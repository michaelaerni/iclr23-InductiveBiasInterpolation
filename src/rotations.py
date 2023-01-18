import argparse
import logging
import typing

import dotenv
import gin
import jax
import kornia
import numpy as np
import torch
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms.functional

import data
import metric_logging
import models
import util

BASE_EXPERIMENT = "rotations"

LOGGING_PRINT_KEYS = (
    "step_loss",
    "step_accuracy",
    "train_loss",
    "train_accuracy",
    "test_accuracy_random",
    "test_grid_invariant_fraction",
    "lr",
)


def main():
    # Prepare environment variables, args, logging, config, seeds
    dotenv.load_dotenv()
    args = parse_args()
    debug = args.debug
    force_cpu = args.cpu
    util.setup_logging(debug=debug)
    config_files = [str(util.CONFIG_DEFAULTS_DIR / f"{BASE_EXPERIMENT}.gin")]
    if args.config is not None:
        config_files.extend(args.config)
    util.setup_gin(config_files=config_files, bindings=args.bindings)

    tags = {"base_experiment": BASE_EXPERIMENT}
    if args.tag is not None:
        tags["tag"] = args.tag
    with metric_logging.MetricLogger(
        experiment_name=args.experiment, run_name=args.run, tags=tags, debug=debug
    ) as metric_logger:
        Rotations(metric_logger=metric_logger, force_cpu=force_cpu).run()


@gin.configurable
class Rotations(object):
    def __init__(
        self,
        metric_logger: metric_logging.MetricLogger,
        force_cpu: bool,
        dataset_loader: typing.Union[data.DatasetLoader, data.SyntheticData],
        convert_grayscale: bool,
        num_train_samples: int,
        num_test_samples: int,
        num_train_steps: int,
        noise_fraction: float,
        num_train_rotations: int,
        stratify_samples: bool,
        standardize_features: bool,
        interpolation_mode: typing.Literal["bilinear", "nearest"],
        crop_size: typing.Optional[int | typing.Tuple[int, int]],
        test_grid_size: int,
        network_builder: models.base.NetworkBuilder,
        train_batch_size: int,
        test_batch_size: int,
        seed: int,
        training_seed: typing.Optional[int],
        log_every: int,
        train_eval_every: int,
        test_eval_every: int,
        test_eval_every_initial: typing.Optional[typing.Tuple[int, int]] = None,
        deterministic_training: bool = True,
    ):
        self.dataset_loader = dataset_loader
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.convert_grayscale = convert_grayscale
        self.num_train_steps = num_train_steps
        self.noise_fraction = noise_fraction
        self.num_train_rotations = num_train_rotations
        self.stratify_samples = stratify_samples
        self.standardize_features = standardize_features
        self.interpolation_mode = interpolation_mode
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        self.test_grid_size = test_grid_size
        self.network_builder = network_builder
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.training_seed = training_seed
        self.log_every = log_every
        self.train_eval_every = train_eval_every
        self.test_eval_every = test_eval_every
        self.test_eval_every_initial = (
            test_eval_every_initial if test_eval_every_initial is not None else (0, 1)
        )
        self._deterministic_training = deterministic_training

        self._log = logging.getLogger(__name__)
        self._metric_logger = metric_logger

        # Currently constant:
        self.dtype = torch.float32
        if not force_cpu and torch.cuda.is_available():
            self._log.info("Training on GPU")
            self.device = torch.device("cuda")
        else:
            self._log.info("Training on CPU")
            self.device = torch.device("cpu")

        config_dict = {
            "dataset": self.dataset_loader.name,
            "num_train_samples": self.num_train_samples,
            "num_test_samples": self.num_test_samples,
            "convert_grayscale": self.convert_grayscale,
            "num_train_steps": self.num_train_steps,
            "noise_fraction": self.noise_fraction,
            "num_train_rotations": self.num_train_rotations,
            "stratify_samples": self.stratify_samples,
            "standardize_features": self.standardize_features,
            "interpolation_mode": self.interpolation_mode,
            "crop_size": self.crop_size,
            "test_grid_size": self.test_grid_size,
            "network_builder": type(network_builder).__name__,
            "train_batch_size": self.train_batch_size,
            "test_batch_size": self.test_batch_size,
            "seed": self.seed,
            "training_seed": self.training_seed,
            "log_every": self.log_every,
            "train_eval_every": self.train_eval_every,
            "test_eval_every": self.test_eval_every,
            "test_eval_every_initial": self.test_eval_every_initial,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "deterministic_training": self._deterministic_training,
        }
        config_dict.update(
            {
                f"network_builder_{key}": value
                for key, value in self.network_builder.config_dict.items()
            }
        )
        config_dict.update(
            {f"data_{key}": value for key, value in self.dataset_loader.config.items()}
        )
        self._metric_logger.log_params(config_dict)

    def run(self):
        if not self._deterministic_training:
            self._log.warning(
                "Using non-deterministic training algorithms; results will not be reproducible!"
            )
        util.setup_seeds(
            self.seed,
            deterministic_algorithms=self._deterministic_training,
            benchmark_algorithms=not self._deterministic_training,  # always benchmark if non-det.
        )

        train_loader, train_eval_loader, test_loader, sample_shape = self.build_datasets()

        # Re-seed training (including model init) if training seed specified
        if self.training_seed is not None:
            torch.manual_seed(self.training_seed)

        model = self.network_builder.build(
            sample_shape=sample_shape,
            num_classes=1 if self.is_binary else self.dataset_loader.num_classes,
            device=self.device,
            dtype=self.dtype,
        )
        num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        self._log.info("Total %d trainable parameters", num_params)

        # Build optimizer and lr scheduler
        optimizer, optimizer_config = self.optimizer(model.parameters())
        scheduler, scheduler_config = self.lr_scheduler(optimizer)
        self._metric_logger.log_params(
            {"num_params": num_params} | optimizer_config | scheduler_config
        )
        loss_fn = (
            torch.nn.BCEWithLogitsLoss(reduction="none")
            if self.is_binary
            else torch.nn.CrossEntropyLoss(reduction="none")
        )

        train_accuracy = None
        test_accuracy = None
        best_test_accuracy = 0.0
        early_stopped_training_accuracy = -1.0
        self._log.info("Starting training")
        for step, (batch_xs, batch_ys) in zip(
            range(1, self.num_train_steps + 1), self._cycle(train_loader)
        ):
            epoch = (step - 1) // len(train_loader) + 1
            log_dict = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],  # lr used DURING this step
            }

            model.train()
            optimizer.zero_grad()
            logits = model(batch_xs.to(self.device))
            batch_loss = loss_fn(logits, batch_ys.to(self.device))
            batch_loss.mean().backward()
            optimizer.step()
            scheduler.step()

            if self.is_binary:
                batch_num_correct = (
                    (torch.round(torch.sigmoid(logits)).cpu() == batch_ys).int().sum().item()
                )
            else:
                batch_num_correct = (
                    (torch.argmax(logits, dim=-1).cpu() == batch_ys).int().sum().item()
                )
            log_dict.update(
                {
                    "step_loss": batch_loss.detach().mean().item(),
                    "step_accuracy": float(batch_num_correct) / batch_xs.size(0),
                }
            )

            # Determine if and what to evaluate
            test_eval_init_steps, test_eval_init_freq = self.test_eval_every_initial
            test_eval_every = (
                test_eval_init_freq if step <= test_eval_init_steps else self.test_eval_every
            )
            do_test_eval = (
                step == 1 or step == self.num_train_steps or (step % test_eval_every == 0)
            )
            do_train_eval = do_test_eval or (step % self.train_eval_every == 0)
            do_log = do_train_eval or do_test_eval or (step % self.log_every == 0)

            if do_train_eval or do_test_eval:
                model.eval()

            # Check training loss for stopping and evaluate
            # Uses same procedure as for training to determine loss and accuracy
            if do_train_eval:
                train_eval_log_dict = self.evaluate_training(
                    model=model,
                    loss=loss_fn,
                    data_loader=train_eval_loader,
                )
                log_dict.update(train_eval_log_dict)
                train_accuracy = train_eval_log_dict["train_accuracy"]

            if do_test_eval:
                # Only evaluate robustness at the end of training
                evaluate_robustness = step == self.num_train_steps
                test_eval_log_dict = self.evaluate_test(
                    model=model,
                    loss=loss_fn,
                    data_loader=test_loader,
                    include_grid=evaluate_robustness,
                )
                log_dict.update(test_eval_log_dict)

                # Test accuracy on random rotations is the quantity of interest
                test_accuracy = log_dict["test_accuracy_random"]
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    assert train_accuracy is not None
                    early_stopped_training_accuracy = train_accuracy

            if do_log:
                self._metric_logger.log_metrics(log_dict, step=step)

                log_quantities = {
                    metric_key: log_dict[metric_key]
                    for metric_key in LOGGING_PRINT_KEYS
                    if metric_key in log_dict
                }
                # FIXME: Could format this more nicely
                self._log.info(
                    "Epoch %d (step %d): %s",
                    epoch,
                    step,
                    ", ".join(f"{key}={value:.4f}" for key, value in log_quantities.items()),
                )
        self._log.info("Finished training")

        assert test_accuracy is not None and train_accuracy is not None
        self._metric_logger.log_metric(
            "best_final_test_accuracy_diff",
            best_test_accuracy - test_accuracy,
            step=self.num_train_steps,
        )
        self._metric_logger.log_metric(
            "best_test_accuracy", best_test_accuracy, step=self.num_train_steps
        )
        assert early_stopped_training_accuracy >= 0.0
        self._metric_logger.log_metric(
            "early_stopped_training_accuracy",
            early_stopped_training_accuracy,
            step=self.num_train_steps,
        )

        self._log.info("Final train accuracy: %.4f", train_accuracy)
        self._log.info("Final test accuracy: %.4f", test_accuracy)
        self._log.info("Done")

    def evaluate_training(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> typing.Dict[str, str]:
        self._log.debug("Evaluating on full training set")

        num_logits = 1 if self.is_binary else self.dataset_loader.num_classes

        train_logits = []
        train_ys = []
        train_noise_mask = []
        for batch_xs, batch_ys, batch_noise_mask in data_loader:
            num_batch_samples, num_grid_rotations = batch_xs.size()[:2]
            pred_logits_grid = torch.empty((num_batch_samples, num_grid_rotations, num_logits))
            for rotation_idx in range(num_grid_rotations):
                pred_logits = model(batch_xs[:, rotation_idx].to(self.device)).detach().cpu()
                pred_logits_grid[:, rotation_idx] = pred_logits
            train_logits.append(pred_logits_grid)

            train_ys.append(batch_ys)
            train_noise_mask.append(batch_noise_mask)
        train_logits = torch.cat(train_logits, dim=0)
        train_ys_true = torch.cat(train_ys, dim=0)
        train_noise_mask = torch.cat(train_noise_mask, dim=0)
        if self.is_binary:
            train_ys_pred_grid = torch.round(torch.sigmoid(train_logits))
        else:
            train_ys_pred_grid = torch.argmax(train_logits, dim=-1)
        assert (
            train_ys_pred_grid.size()
            == (train_ys_true.size(0), self.num_train_rotations) + train_ys_true.size()[1:]
        )
        train_loss_grid = torch.empty(train_logits.size()[:2])
        train_correct_grid = torch.empty(train_logits.size()[:2], dtype=torch.bool)
        for rotation_idx in range(train_logits.size(1)):
            current_loss = loss(train_logits[:, rotation_idx], train_ys_true)
            current_correct = train_ys_true == train_ys_pred_grid[:, rotation_idx]
            if self.is_binary:
                current_loss = current_loss.squeeze(1)
                current_correct = current_correct.squeeze(1)
            train_loss_grid[:, rotation_idx] = current_loss
            train_correct_grid[:, rotation_idx] = current_correct
        train_loss = torch.mean(train_loss_grid).item()
        train_accuracy = torch.mean(train_correct_grid.float()).item()
        train_loss_original = train_loss_grid[:, 0].mean().item()
        train_accuracy_original = torch.mean(train_correct_grid[:, 0].float()).item()
        log_dict = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_loss_original": train_loss_original,
            "train_accuracy_original": train_accuracy_original,
        }
        train_noise_correct_grid = train_correct_grid[train_noise_mask]
        if train_noise_correct_grid.size(0) > 0:
            train_noise_accuracy = torch.mean(train_noise_correct_grid.float()).item()
            log_dict["train_noise_accuracy"] = train_noise_accuracy
        train_noise_correct_original = train_noise_correct_grid[:, 0]
        if train_noise_correct_original.size(0) > 0:
            train_noise_accuracy_original = torch.mean(
                train_noise_correct_original.float()
            ).item()
            log_dict["train_noise_accuracy_original"] = train_noise_accuracy_original

        return log_dict

    def evaluate_test(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        include_grid: bool,
    ) -> typing.Dict[str, str]:
        self._log.debug("Evaluating on full test set")

        log_dict = dict()

        num_logits = 1 if self.is_binary else self.dataset_loader.num_classes

        # Each test xs has one entry for natural, one for random rotation, and rest grid
        test_logits_natural = []
        test_logits_random = []
        test_logits_grid = []
        test_ys = []
        for batch_xs_natural, batch_xs_random, batch_xs_grid, batch_ys in data_loader:
            test_ys.append(batch_ys)

            # Natural
            pred_logits_natural = model(batch_xs_natural.to(self.device)).detach().cpu()
            test_logits_natural.append(pred_logits_natural)

            # Single random rotation
            pred_logits_random = model(batch_xs_random.to(self.device)).detach().cpu()
            test_logits_random.append(pred_logits_random)

            # Grid
            if include_grid:
                num_batch_samples, num_grid_rotations = batch_xs_grid.size()[:2]
                pred_logits_grid = torch.empty(
                    (num_batch_samples, num_grid_rotations, num_logits)
                )
                for rotation_idx in range(num_grid_rotations):
                    pred_logits = (
                        model(batch_xs_grid[:, rotation_idx].to(self.device)).detach().cpu()
                    )
                    pred_logits_grid[:, rotation_idx] = pred_logits
                test_logits_grid.append(pred_logits_grid)
        test_logits_natural = torch.cat(test_logits_natural, dim=0)
        test_logits_random = torch.cat(test_logits_random, dim=0)
        test_ys_true = torch.cat(test_ys, dim=0)
        assert test_logits_random.size() == test_logits_natural.size()

        if num_logits == 1:
            test_ys_pred_natural = torch.round(torch.sigmoid(test_logits_natural)).long()
            test_ys_pred_random = torch.round(torch.sigmoid(test_logits_random)).long()
        else:
            test_ys_pred_natural = torch.argmax(test_logits_natural, dim=-1)
            test_ys_pred_random = torch.argmax(test_logits_random, dim=-1)

        assert test_ys_pred_natural.size() == test_ys_true.size()
        test_loss_natural = loss(test_logits_natural, test_ys_true).mean().item()
        test_accuracy_natural = torch.mean((test_ys_pred_natural == test_ys_true).float()).item()
        log_dict.update(
            {
                "test_loss_natural": test_loss_natural,
                "test_accuracy_natural": test_accuracy_natural,
            }
        )

        # Randomly rotated test samples
        assert test_ys_pred_random.size() == test_ys_true.size()
        test_loss_random = loss(test_logits_random, test_ys_true).mean().item()
        test_accuracy_random = torch.mean((test_ys_pred_random == test_ys_true).float()).item()
        log_dict.update(
            {
                "test_loss_random": test_loss_random,
                "test_accuracy_random": test_accuracy_random,
            }
        )

        # Grid of test sample rotations
        if include_grid:
            test_logits_grid = torch.cat(test_logits_grid, dim=0)
            assert test_logits_grid.size() == (
                test_logits_natural.size(0),
                self.test_grid_size,
                num_logits,
            )
            if num_logits == 1:
                test_ys_pred_grid = torch.round(torch.sigmoid(test_logits_grid)).long()
            else:
                test_ys_pred_grid = torch.argmax(test_logits_grid, dim=-1)

            #  Accuracy on full grid and worst-case-rotation (adversarial/robust)
            test_grid_correct = torch.eq(test_ys_pred_grid, test_ys_true.unsqueeze(1))
            test_accuracy_grid = torch.mean(test_grid_correct.float()).item()
            test_accuracy_robust = torch.mean(torch.all(test_grid_correct, dim=-1).float()).item()
            log_dict.update(
                {
                    "test_accuracy_grid": test_accuracy_grid,
                    "test_accuracy_robust": test_accuracy_robust,
                }
            )

            #  Fraction of invariantly-predicted test samples
            if num_logits == 1:
                test_ys_pred_grid = test_ys_pred_grid.squeeze(-1)
            assert test_ys_pred_grid.size() == (test_ys_true.size(0), self.test_grid_size)
            max_same_class_per_sample = torch.zeros((test_ys_true.size(0),), dtype=torch.long)
            for sample_idx in range(test_ys_true.size(0)):
                current_counts = torch.bincount(
                    test_ys_pred_grid[sample_idx],
                    minlength=(2 if self.is_binary else self.dataset_loader.num_classes),
                )
                max_same_class_per_sample[sample_idx] = current_counts.max()
            test_grid_invariant_fraction = torch.mean(
                (max_same_class_per_sample == test_ys_pred_grid.size(1)).float()
            ).item()
            test_grid_average_agreement = torch.mean(
                max_same_class_per_sample / float(test_ys_pred_grid.size(1))
            ).item()
            log_dict.update(
                {
                    "test_grid_invariant_fraction": test_grid_invariant_fraction,
                    "test_grid_average_agreement": test_grid_average_agreement,
                }
            )

        return log_dict

    def build_datasets(
        self,
    ) -> typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.Size,
    ]:
        if isinstance(self.dataset_loader, data.DatasetLoader):
            all_xs, all_ys = self.dataset_loader.load_dataset()
            all_xs = all_xs.to(self.dtype)

            if self.convert_grayscale and all_xs.size(1) == 3:
                all_xs = kornia.color.rgb_to_grayscale(all_xs)

            all_ys = all_ys.long()
            num_classes = self.dataset_loader.num_classes

            if not self.stratify_samples:
                all_indices_shuffled = torch.randperm(all_xs.size(0))

                # Take test indices first to have same test set for different train set sizes
                test_indices = all_indices_shuffled[: self.num_test_samples]
                train_indices = all_indices_shuffled[
                    self.num_test_samples : self.num_test_samples + self.num_train_samples
                ]
                assert train_indices.size() == (
                    self.num_train_samples,
                ) and test_indices.size() == (self.num_test_samples,)
            else:
                # Stratified split into training and test data
                train_indices = []
                test_indices = []
                num_train_samples_per_class = self.num_train_samples // num_classes
                num_test_samples_per_class = self.num_test_samples // num_classes
                for original_class in range(num_classes):
                    class_indices = torch.argwhere((all_ys == original_class)).squeeze()
                    shuffled_indices = torch.randperm(class_indices.size(0))

                    # Take indices from different ends to have same set if other changes
                    assert (
                        num_train_samples_per_class + num_test_samples_per_class
                        <= class_indices.size(0)
                    )
                    test_indices.append(
                        class_indices[shuffled_indices[:num_test_samples_per_class]]
                    )
                    train_indices.append(
                        class_indices[shuffled_indices[-num_train_samples_per_class:]]
                    )

                train_indices = torch.cat(train_indices)
                test_indices = torch.cat(test_indices)
                assert train_indices.size() == (num_classes * num_train_samples_per_class,)
                assert test_indices.size() == (num_classes * num_test_samples_per_class,)

            test_xs_original = all_xs[test_indices]
            test_ys = all_ys[test_indices]
            train_xs_original = all_xs[train_indices]
            train_ys_clean = all_ys[train_indices]
        else:
            assert isinstance(self.dataset_loader, data.SyntheticData)
            assert not self.convert_grayscale, "Grayscale for synthetic data not supported"
            with jax.default_device(jax.devices("cpu")[0]):
                key = jax.random.PRNGKey(self.seed)
                test_sample_key, train_sample_key = jax.random.split(key)
                test_xs_original_jax, test_ys_jax = self.dataset_loader.generate_samples(
                    num_samples=self.num_test_samples,
                    key=test_sample_key,
                )
                train_xs_original_jax, train_ys_clean_jax = self.dataset_loader.generate_samples(
                    num_samples=self.num_train_samples,
                    key=train_sample_key,
                )

                # Move everything from JAX to PyTorch
                test_xs_original = torch.from_numpy(
                    np.asarray(test_xs_original_jax.transpose(0, 3, 1, 2))
                ).to(self.dtype)
                test_ys = torch.from_numpy(np.asarray(test_ys_jax)).squeeze().long()
                train_xs_original = torch.from_numpy(
                    np.asarray(train_xs_original_jax.transpose(0, 3, 1, 2))
                ).to(self.dtype)
                train_ys_clean = torch.from_numpy(np.asarray(train_ys_clean_jax)).squeeze().long()

                del test_xs_original_jax, test_ys_jax
                del train_xs_original_jax, train_ys_clean_jax

                all_xs = torch.concat((train_xs_original, test_xs_original))
                all_ys = torch.concat((train_ys_clean, test_ys))

        assert all_xs.dim() == 4 and all_ys.dim() == 1 and all_ys.size(0) == all_xs.size(0)
        assert all_ys.min().item() == 0
        assert all_ys.max().item() == (
            1 if self.is_binary else self.dataset_loader.num_classes - 1
        )

        if self.is_binary:
            train_ys_clean = torch.unsqueeze(train_ys_clean, -1).to(self.dtype)
            test_ys = torch.unsqueeze(test_ys, -1).to(self.dtype)

        # Standardize channel values
        if self.standardize_features:
            # Standardize original images to have clean rotations
            channels_std, channels_mean = torch.std_mean(
                train_xs_original, dim=(0, 2, 3), unbiased=True, keepdim=True
            )
            assert (
                channels_mean.size() == (1, all_xs.size(1), 1, 1)
                and channels_std.size() == channels_mean.size()
            )
            train_xs_original = (train_xs_original - channels_mean) / channels_std
            test_xs_original = (test_xs_original - channels_mean) / channels_std

        # Calculate crop target size if cropping is enabled
        crop_params = None
        if self.crop_size is not None:
            original_height, original_width = all_xs.size()[-2:]
            crop_height, crop_width = self.crop_size
            assert crop_height <= original_height and crop_width <= original_width
            crop_top = original_height // 2 - crop_height // 2
            crop_left = original_width // 2 - crop_width // 2
            crop_params = (crop_top, crop_left, crop_height, crop_width)

        # Transform test data
        # Use clean, random rotation, and grid rotations
        test_xs_random, test_xs_grid = self._transform_test_data(test_xs_original, crop_params)
        if crop_params is not None:
            test_xs_original = torchvision.transforms.functional.crop(
                test_xs_original, *crop_params
            )

        assert test_xs_random.size() == test_xs_original.size()
        assert (
            test_xs_grid.size()
            == (test_xs_original.size(0), self.test_grid_size) + test_xs_original.size()[1:]
        )

        # Leave everything on CPU due to rotations, and move to GPU ad-hoc

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                test_xs_original, test_xs_random, test_xs_grid, test_ys
            ),
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Randomly rotate training samples; stays fixed THROUGHOUT TRAINING
        train_xs_rotated = self._transform_train_data(train_xs_original, crop_params)
        assert (
            train_xs_rotated.size(0) == train_xs_original.size(0)
            and train_xs_rotated.size(1) == self.num_train_rotations
        )
        assert train_xs_rotated.size()[2:] == test_xs_grid.size()[2:]

        # Apply noise to training labels
        # Do this at the very end to ensure that train xs w/ and w/o noise are exactly the same
        num_noise_labels = int(train_ys_clean.size(0) * self.noise_fraction)
        assert 0 <= num_noise_labels <= train_xs_original.size(0)
        train_noise_mask = torch.zeros(train_ys_clean.size(0), dtype=torch.bool)
        if num_noise_labels > 0:
            train_ys_noise = torch.clone(train_ys_clean)
            noise_indices = torch.randperm(train_ys_noise.size(0))[:num_noise_labels]
            if self.is_binary:
                train_ys_noise[noise_indices] = 1.0 - train_ys_clean[noise_indices]
            else:
                noise_labels_raw = torch.randint(
                    0, self.dataset_loader.num_classes - 1, size=(num_noise_labels,)
                )
                train_ys_noise[noise_indices] = torch.where(
                    noise_labels_raw < train_ys_noise[noise_indices],
                    noise_labels_raw,
                    noise_labels_raw + 1,
                )

            # noinspection PyTypeChecker
            assert torch.all(train_ys_clean[noise_indices] != train_ys_noise[noise_indices])
            train_ys = train_ys_noise
            train_noise_mask[noise_indices] = True
        else:
            train_ys = train_ys_clean

        # Flatten training samples for training, but keep with grid info for test
        train_xs_training = torch.flatten(train_xs_rotated, 0, 1)
        train_ys_training = torch.repeat_interleave(train_ys, self.num_train_rotations, dim=0)
        assert train_xs_training.size(0) == train_ys_training.size(0)
        assert train_xs_training.size()[1:] == train_xs_rotated.size()[2:]
        assert train_ys_training.size()[1:] == train_ys.size()[1:]

        self._log.info("Using %d raw training samples", train_xs_original.size(0))
        self._log.info("Using %d raw test samples", test_xs_original.size(0))
        self._log.info("Using %d actual training samples", train_xs_training.size(0))
        self._metric_logger.log_params(
            {
                "raw_num_train_samples": train_xs_original.size(0),
                "raw_num_test_samples": test_xs_original.size(0),
                "effective_num_train_samples": train_xs_training.size(0),
            }
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_xs_training.detach(), train_ys_training),
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=False,
        )

        train_eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_xs_rotated.detach(), train_ys, train_noise_mask),
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )

        sample_shape = all_xs.size()[1:]
        return train_loader, train_eval_loader, test_loader, sample_shape

    def _transform_test_data(
        self,
        test_xs_original: torch.Tensor,
        crop_params: typing.Optional[typing.Tuple[int, int, int, int]],
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        target_height, target_width = (
            crop_params[-2:] if crop_params is not None else test_xs_original.size()[-2:]
        )

        test_random_rotations = torch.rand((test_xs_original.size(0),)) * 360.0
        test_xs_random = kornia.geometry.rotate(
            test_xs_original,
            angle=test_random_rotations,
            mode=self.interpolation_mode,
        )
        if crop_params is not None:
            test_xs_random = torchvision.transforms.functional.crop(test_xs_random, *crop_params)
        # Cannot use grid from -360 to 360 as that includes 360 twice
        test_grid_angles = torch.linspace(
            0.0, 360.0 - (360.0 / self.test_grid_size), steps=self.test_grid_size
        )

        test_xs_grid = torch.empty(
            (
                test_xs_original.size(0),
                test_grid_angles.size(0),
                test_xs_original.size(1),
                target_height,
                target_width,
            ),
            dtype=self.dtype,
        )
        for rotation_idx, current_angle in enumerate(test_grid_angles):
            rotated_xs = kornia.geometry.rotate(
                test_xs_original,
                angle=current_angle,
                mode=self.interpolation_mode,
            )
            if crop_params is not None:
                rotated_xs = torchvision.transforms.functional.crop(rotated_xs, *crop_params)
            test_xs_grid[:, rotation_idx] = rotated_xs

        return test_xs_random, test_xs_grid

    def _transform_train_data(
        self,
        train_xs_original: torch.Tensor,
        crop_params: typing.Optional[typing.Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        assert self.num_train_rotations > 0
        target_height, target_width = (
            crop_params[-2:] if crop_params is not None else train_xs_original.size()[-2:]
        )
        train_xs_rotated = torch.empty(
            (
                train_xs_original.size(0),
                self.num_train_rotations,
                train_xs_original.size(1),
                target_height,
                target_width,
            ),
            dtype=self.dtype,
        )

        # Cannot use grid from -360 to 360 as that includes 360 twice
        train_angles_grid = torch.linspace(
            0.0,
            360.0 - (360.0 / self.num_train_rotations),
            steps=self.num_train_rotations,
        )
        # Randomly rotate each training sample with a different angle
        # Avoids potential distribution shift effects
        angle_offsets = torch.rand((train_xs_original.size(0),)) * 360.0
        train_xs_angles = angle_offsets.unsqueeze(-1) + torch.tile(
            train_angles_grid, dims=(angle_offsets.size(0), 1)
        )

        assert train_xs_angles.size() == (train_xs_original.size(0), self.num_train_rotations)
        for sample_idx in range(train_xs_original.size(0)):
            current_x_tiled = torch.tile(
                train_xs_original[sample_idx], dims=(self.num_train_rotations, 1, 1, 1)
            )
            current_xs_rotated = kornia.geometry.rotate(
                current_x_tiled,
                angle=train_xs_angles[sample_idx],
                mode=self.interpolation_mode,
            )
            assert (
                current_xs_rotated.size()
                == (self.num_train_rotations,) + train_xs_original.size()[1:]
            )
            if crop_params is not None:
                current_xs_rotated = torchvision.transforms.functional.crop(
                    current_xs_rotated, *crop_params
                )
            train_xs_rotated[sample_idx] = current_xs_rotated

        return train_xs_rotated

    @gin.configurable(denylist=("params",))
    def optimizer(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        cls: typing.Type[torch.optim.Optimizer],
        **optimizer_kwargs,
    ) -> typing.Tuple[torch.optim.Optimizer, typing.Dict[str, typing.AnyStr]]:
        optimizer = cls(params, **optimizer_kwargs)
        merged_args = optimizer.defaults | optimizer_kwargs
        config_dict = {f"optimizer_{key}": value for key, value in merged_args.items()}
        config_dict["optimizer"] = cls.__name__
        return optimizer, config_dict

    @gin.configurable(denylist=("optimizer",))
    def lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        cls: typing.Optional[typing.Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        **scheduler_kwargs,
    ) -> typing.Tuple[torch.optim.lr_scheduler._LRScheduler, typing.Dict[str, typing.AnyStr]]:
        if cls is None:
            if len(scheduler_kwargs) > 0:
                raise ValueError("Cannot supply scheduler arguments without scheduler")
            cls = torch.optim.lr_scheduler.ConstantLR
            scheduler_kwargs = {"factor": 1.0}
        scheduler = cls(optimizer, **scheduler_kwargs)
        config_dict = {f"scheduler_{key}": value for key, value in scheduler_kwargs.items()}
        config_dict["scheduler"] = cls.__name__
        return scheduler, config_dict

    @property
    def is_binary(self) -> bool:
        return self.dataset_loader.num_classes == 2

    @staticmethod
    def _cycle(loader: torch.utils.data.DataLoader):
        while True:
            for batch in loader:
                yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Gin configuration sources
    parser.add_argument("--config", nargs="*", type=str, help="Gin configuration files")
    parser.add_argument("--bindings", nargs="*", type=str, help="Additional gin bindings")

    # Other experiment-specific arguments
    parser.add_argument("--tag", type=str, required=False, help="Optional tag for run")
    parser.add_argument("--run", type=str, required=False, help="Optional run name")
    parser.add_argument("--experiment", type=str, required=False, help="Experiment name")
    parser.add_argument("--debug", action="store_true", help="Enable debug functionality")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU available")

    return parser.parse_args()


if __name__ == "__main__":
    main()
