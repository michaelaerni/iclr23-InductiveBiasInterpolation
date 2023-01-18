import logging
import math
import os
import pathlib
import random
import typing

import gin.torch.external_configurables  # automatically registers many torch modules
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import torch
import torch.backends.cudnn
import torch.utils.data

# Important directories
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DEFAULT_DATASET_DIR = ROOT_DIR / "datasets"
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DEFAULTS_DIR = CONFIG_DIR / "defaults"


def setup_gin(
    config_files: typing.Sequence[str],
    bindings: typing.Optional[typing.Sequence[str]],
):
    # Can add additional (e.g., torch-specific) classes as follows:
    gin.config.external_configurable(
        torch.optim.lr_scheduler.LinearLR, module="torch.optim.lr_scheduler"
    )
    gin.config.external_configurable(optax.warmup_cosine_decay_schedule, module="optax")
    gin.config.external_configurable(optax.warmup_exponential_decay_schedule, module="optax")
    gin.config.external_configurable(optax.constant_schedule, module="optax")
    gin.config.external_configurable(optax.piecewise_constant_schedule, module="optax")
    gin.config.external_configurable(optax.sgd, module="optax")
    gin.config.external_configurable(optax.adam, module="optax")

    gin.parse_config_files_and_bindings(
        config_files=config_files,
        bindings=bindings,
        skip_unknown=False,
        print_includes_and_imports=False,
    )


def setup_logging(debug: bool):
    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format="{asctime} [{levelname}] ({name}): {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=log_level,
        force=True,  # to avoid conflicts with absl logger
    )

    # Set level to warning for known verbose modules
    # Can uncomment the following lines to make logging output less verbose
    # logging.getLogger("PIL").setLevel(logging.WARNING)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.INFO if debug else logging.WARNING)


def setup_seeds(
    seed: int,
    deterministic_algorithms: bool = True,
    benchmark_algorithms: bool = False,
):
    # Globally fix seeds in case manual seeding is missing somewhere
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic_algorithms:
        # Enable deterministic (GPU) operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        if benchmark_algorithms:
            raise ValueError("Benchmarking should not be enabled under deterministic algorithms")

    # NB: benchmarking significantly improves training in general,
    #  but can reduce performance if things like input shapes change a lot!
    torch.backends.cudnn.benchmark = benchmark_algorithms


class JAXDataLoader(object):
    def __init__(
        self,
        *arrays: jnp.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        assert all(array.shape[0] == arrays[0].shape[0] for array in arrays[1:])
        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(self.arrays[0].shape[0] / self.batch_size)

    def epoch(
        self,
        shuffle_key: typing.Optional[jax.random.KeyArray] = None,
    ) -> typing.Iterator[typing.Tuple[jnp.ndarray, ...]]:
        num_samples = self.arrays[0].shape[0]
        if self.shuffle:
            assert shuffle_key is not None
            target_order = jax.random.permutation(shuffle_key, num_samples)
        else:
            target_order = jnp.arange(num_samples)

        for batch_idx in range(len(self)):
            batch_offset = batch_idx * self.batch_size
            batch_indices = target_order[
                batch_offset : min(batch_offset + self.batch_size, num_samples)
            ]
            yield tuple(jax.lax.stop_gradient(array[batch_indices]) for array in self.arrays)


@gin.configurable(denylist=("optimizer",))
class WarmupScheduler(torch.optim.lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        scheduler_cls: typing.Type[torch.optim.lr_scheduler._LRScheduler],
        keep_epoch: bool = True,
        **scheduler_kwargs,
    ):
        scheduler = scheduler_cls(optimizer=optimizer, **scheduler_kwargs)

        if warmup_steps > 0:
            # Can assume warmup_steps > 0, else this will never be called
            # Step >= 0
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: (step + 1) / warmup_steps
            )

            super(WarmupScheduler, self).__init__(
                optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps]
            )
        else:
            super(WarmupScheduler, self).__init__(
                optimizer, schedulers=[scheduler], milestones=[]
            )

        self._keep_epoch = keep_epoch

    def step(self, **kwargs):
        # Adjusted from normal sequential scheduler to continue with epoch instead of resetting
        self.last_epoch += 1
        if self.last_epoch < self._milestones[0] or len(self._schedulers) == 1:
            idx = 0
        else:
            idx = 1
        if len(self._milestones) > 0 and self.last_epoch == self._milestones[0]:
            self._schedulers[idx].step(self.last_epoch if self._keep_epoch else 0)
        else:
            self._schedulers[idx].step()

        self._last_lr = self._schedulers[idx].get_last_lr()


@gin.configurable(denylist=("optimizer",))
class InverseSqrtLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_rate: int,
        discrete_steps: bool = False,
        last_epoch: int = -1,
    ):
        def lr_lambda(step: int) -> float:
            factor = step / decay_rate
            if discrete_steps:
                factor = math.floor(factor)
            return 1.0 / math.sqrt(1.0 + factor)

        super(InverseSqrtLR, self).__init__(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch,
        )


@gin.configurable
def warmup_inverse_sqrt_schedule(
    peak_value: float,
    transition_steps: int | float,
    warmup_steps: int = 0,
    init_value: float = 1e-6,
    staircase: bool = False,
) -> optax.Schedule:
    assert warmup_steps >= 0
    assert transition_steps > 0

    def decay_schedule(count):
        factor = count / transition_steps
        if staircase:
            factor = jnp.floor(factor)
        return peak_value / jnp.sqrt(1.0 + factor)

    schedules = (
        optax.linear_schedule(
            init_value=init_value, end_value=peak_value, transition_steps=warmup_steps
        ),
        decay_schedule,
    )
    return optax.join_schedules(schedules, (warmup_steps,))


def get_data_dir() -> pathlib.Path:
    env_data_dir = os.environ.get("DATA_DIR", default=None)
    if env_data_dir:
        return pathlib.Path(env_data_dir)
    else:
        return DEFAULT_DATASET_DIR
