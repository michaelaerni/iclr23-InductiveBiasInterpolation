import argparse
import logging
import math
import typing

import chex  # FIXME: include in conda env with fixed version
import dotenv
import flax
import flax.traverse_util
import gin
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

import data
import metric_logging
# noinspection PyUnresolvedReferences
import models  # required for gin to find models
import util

BASE_EXPERIMENT = "filter_size"

LOGGING_PRINT_KEYS = (
    "epoch_loss",
    "train_loss",
    "train_accuracy",
    "test_accuracy",
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
        FilterExperiment(metric_logger=metric_logger, force_cpu=force_cpu).run()


@gin.configurable
class FilterExperiment(object):
    def __init__(
        self,
        metric_logger: metric_logging.MetricLogger,
        force_cpu: bool,
        dataset: data.SyntheticData | data.DatasetLoader,
        num_train_samples: int,
        num_test_samples: int,
        binarized_classes: typing.Optional[
            typing.Tuple[typing.Tuple[int, ...], typing.Tuple[typing.Tuple[int, ...]]]
        ],
        convert_grayscale: bool,
        standardize_features: bool,
        crop_size: typing.Optional[int | typing.Tuple[int, int]],
        train_epochs: int,
        noise_fraction: float,
        train_batch_size: int,
        test_batch_size: int,
        seed: int,
        log_every: int,
        train_eval_every: int,
        test_eval_every: int,
        test_eval_every_initial: typing.Optional[typing.Tuple[int, int]] = None,
        training_seed: typing.Optional[int] = None,
        deterministic_training: bool = True,
        plot_filters: bool = False,
    ):
        self.dataset = dataset
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.binarized_classes = binarized_classes
        self.convert_grayscale = convert_grayscale
        self.standardize_features = standardize_features
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        self.train_epochs = train_epochs
        self.noise_fraction = noise_fraction
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
        self.deterministic_training = deterministic_training
        self._plot_filters = plot_filters

        self._log = logging.getLogger(self.__class__.__name__)
        self._metric_logger = metric_logger

        # Currently constant:
        self.dtype = jnp.float32
        if not force_cpu:
            self.device = jax.devices()[0]
            self._log.info("Training on default device (%s)", self.device)
        else:
            self._log.info("Training on CPU")
            self.device = jax.devices("cpu")[0]
            raise NotImplementedError("This does most likely not work yet")

        config_dict = {
            "dataset": self.dataset.name,
            "num_train_samples": self.num_train_samples,
            "num_test_samples": self.num_test_samples,
            "binarized_classes": str(self.binarized_classes),
            "convert_grayscale": self.convert_grayscale,
            "standardize_features": self.standardize_features,
            "crop_size": str(self.crop_size),
            "train_epochs": self.train_epochs,
            "noise_fraction": self.noise_fraction,
            "train_batch_size": self.train_batch_size,
            "test_batch_size": self.test_batch_size,
            "seed": self.seed,
            "log_every": self.log_every,
            "train_eval_every": self.train_eval_every,
            "test_eval_every": self.test_eval_every,
            "test_eval_every_initial": self.test_eval_every_initial,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "training_seed": self.training_seed,
            "deterministic_training": self.deterministic_training,
            "plot_filters": self._plot_filters,
        }
        config_dict.update({f"data_{key}": value for key, value in self.dataset.config.items()})
        self._metric_logger.log_params(config_dict)
        additional_tags = self.additional_tags()
        if len(additional_tags) > 0:
            self._metric_logger.log_tags(additional_tags)

    def run(self):
        if not self.deterministic_training:
            self._log.warning(
                "Using non-deterministic training algorithms; results will not be reproducible!"
            )
        util.setup_seeds(
            self.seed,
            deterministic_algorithms=self.deterministic_training,
            benchmark_algorithms=not self.deterministic_training,  # always benchmark if non-det.
        )
        key = jax.random.PRNGKey(self.seed)

        dataset_key, key = jax.random.split(key)
        # Process and store dataset on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            train_loader, train_eval_loader, test_loader, batch_shape = self.build_datasets(
                dataset_key
            )

        # Build and initialize network
        # NHWC !!
        model, model_log_dict = self.network(
            num_classes=1,
            dtype=self.dtype,
        )
        if self.training_seed is None:
            init_key, shuffle_key = jax.random.split(key)
        else:
            # Use custom seed for NN init and training
            # This allows to average over training runs with same dataset
            init_key, shuffle_key = jax.random.split(jax.random.PRNGKey(self.training_seed))
        del key  # make sure old key is not used anymore
        variables = model.init(init_key, jnp.ones(batch_shape, dtype=self.dtype))
        state, params = variables.pop("params")
        del variables

        num_params = sum(
            math.prod(shape)
            for shape in jax.tree_util.tree_map(
                jnp.shape, flax.traverse_util.flatten_dict(params)
            ).values()
        )
        self._log.info("Total %d trainable parameters", num_params)
        self._metric_logger.log_params({"num_params": num_params})

        # Build optimizer
        lr_schedule, schedule_log_dict = self.lr_schedule(steps_per_epoch=len(train_loader))
        optimizer, optimizer_log_dict = self.optimizer(lr_schedule=lr_schedule)
        self._metric_logger.log_params(model_log_dict | schedule_log_dict | optimizer_log_dict)
        opt_state = optimizer.init(params)

        def loss_fn(logits: jnp.ndarray, ys: jnp.ndarray):
            # ys are {0, 1} with shape (n, 1), logits have shape (n, 1)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels=ys))

        @jax.jit
        def train_step(
            params: flax.core.FrozenDict,
            opt_state,
            state: flax.core.FrozenDict,
            batch_xs,
            batch_ys,
        ):
            def train_loss_fn(params: flax.core.FrozenDict):
                logits, updated_state = model.apply(
                    {"params": params, **state}, batch_xs, mutable=tuple(state.keys())
                )
                loss = loss_fn(logits=logits, ys=batch_ys)
                return loss, (logits, updated_state)

            grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
            (batch_loss, (pred_logits, state)), grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, state, batch_loss, pred_logits

        train_accuracy = None
        test_accuracy = None
        best_test_accuracy = 0.0
        epoch = 0  # required for later
        self._log.info("Starting training")
        step = 1
        for epoch in range(1, self.train_epochs + 1):
            log_dict = {"epoch": epoch}

            epoch_loss = 0.0
            epoch_num_correct = 0
            epoch_num_samples = 0
            current_shuffle_key, shuffle_key = jax.random.split(shuffle_key)
            for batch_xs, batch_ys in train_loader.epoch(current_shuffle_key):
                batch_xs = jax.device_put(batch_xs, device=self.device)
                batch_ys = jax.device_put(batch_ys, device=self.device)
                params, opt_state, state, batch_loss, batch_logits = train_step(
                    params=params,
                    opt_state=opt_state,
                    state=state,
                    batch_xs=batch_xs,
                    batch_ys=batch_ys,
                )

                epoch_loss += float(batch_loss) * batch_xs.shape[0]
                epoch_num_correct += int(
                    jnp.sum(jnp.round(jax.nn.sigmoid(batch_logits)) == batch_ys).astype(jnp.int32)
                )
                epoch_num_samples += batch_xs.shape[0]
                step += 1

            log_dict.update(
                {
                    "lr": float(lr_schedule(step - 1)),
                    "epoch_accuracy": float(epoch_num_correct) / epoch_num_samples,
                    "epoch_loss": epoch_loss / epoch_num_samples,
                }
            )

            # Determine what to evaluate
            test_eval_init_steps, test_eval_init_freq = self.test_eval_every_initial
            test_eval_every = (
                test_eval_init_freq if epoch <= test_eval_init_steps else self.test_eval_every
            )
            do_test_eval = (
                epoch == 1 or epoch == self.train_epochs or (epoch % test_eval_every == 0)
            )
            do_train_eval = do_test_eval or (epoch % self.train_eval_every == 0)
            do_log = do_train_eval or do_test_eval or (epoch % self.log_every == 0)

            predict_fn = None
            if do_train_eval or do_test_eval:

                @jax.jit
                def predict_fn(xs: jnp.array) -> jnp.array:
                    return model.apply({"params": params, **state}, xs)

            # Check training loss for stopping and evaluate
            # Uses same procedure as for training to determine loss and accuracy
            if do_train_eval:
                self._log.debug("Evaluating on full training set")
                assert predict_fn is not None
                train_eval_log_dict = self.evaluate_model(
                    predict_fn=predict_fn,
                    loss_fn=loss_fn,
                    data_loader=train_eval_loader,
                    noise=True,
                )
                log_dict.update(
                    {f"train_{key}": value for key, value in train_eval_log_dict.items()}
                )
                train_accuracy = train_eval_log_dict["accuracy"]

            if do_test_eval:
                self._log.debug("Evaluating on test set")
                assert predict_fn is not None
                test_eval_log_dict = self.evaluate_model(
                    predict_fn=predict_fn,
                    loss_fn=loss_fn,
                    data_loader=test_loader,
                    noise=False,
                )
                log_dict.update(
                    {f"test_{key}": value for key, value in test_eval_log_dict.items()}
                )

                test_accuracy = test_eval_log_dict["accuracy"]
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy

                if self._plot_filters and epoch == self.train_epochs:
                    self._plot_conv_filters(params)

            if do_log or self._log.isEnabledFor(logging.DEBUG):
                self._metric_logger.log_metrics(log_dict, step=epoch)

                log_quantities = {
                    metric_key: log_dict[metric_key]
                    for metric_key in LOGGING_PRINT_KEYS
                    if metric_key in log_dict
                }
                # FIXME: Could format this more nicely
                self._log.info(
                    "Epoch %d: %s",
                    epoch,
                    ", ".join(f"{key}={value:.4f}" for key, value in log_quantities.items()),
                )

        self._log.info("Finished training")

        assert test_accuracy is not None and train_accuracy is not None
        self._metric_logger.log_metric(
            "best_final_test_accuracy_diff", best_test_accuracy - test_accuracy, step=epoch
        )
        self._metric_logger.log_metric("best_test_accuracy", best_test_accuracy, step=epoch)

        self._log.info("Final train accuracy: %.8f", train_accuracy)
        self._log.info("Final test accuracy: %.8f", test_accuracy)
        self._log.info("Done")

    def evaluate_model(
        self,
        predict_fn: typing.Callable[[jnp.array], jnp.array],
        loss_fn: typing.Callable[[jnp.array, jnp.array], jnp.array],
        data_loader: util.JAXDataLoader,
        noise: bool,
    ) -> typing.Dict[str, typing.Any]:
        logits = []
        ys_true = []
        noise_mask = []
        for batch in data_loader.epoch():
            batch_xs, batch_ys = batch[:2]
            batch_xs = jax.device_put(batch_xs, device=self.device)

            batch_logits = predict_fn(batch_xs)
            batch_logits = jax.device_put(batch_logits, device=batch_ys.device())

            logits.append(batch_logits)
            ys_true.append(batch_ys)
            if noise:
                (batch_noise_mask,) = batch[2:]
                noise_mask.append(batch_noise_mask)

        logits = jnp.concatenate(logits, axis=0)
        ys_true = jnp.concatenate(ys_true, axis=0)

        ys_pred = jnp.round(jax.nn.sigmoid(logits))
        assert ys_true.shape == ys_pred.shape
        pred_correct = jnp.equal(ys_true, ys_pred)
        accuracy = jnp.mean(pred_correct.astype(jnp.float32))
        loss = loss_fn(logits, ys_true)
        log_dict = {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
        if noise:
            noise_mask = jnp.concatenate(noise_mask, axis=0)
            pred_noise_correct = pred_correct[noise_mask]
            if pred_noise_correct.shape[0] > 0:
                noise_accuracy = jnp.mean(pred_noise_correct.astype(jnp.float32))
                log_dict["noise_accuracy"] = float(noise_accuracy)

        return log_dict

    def build_datasets(
        self,
        key: jax.random.KeyArray,
    ) -> typing.Tuple[
        util.JAXDataLoader,
        util.JAXDataLoader,
        util.JAXDataLoader,
        typing.Tuple[int, ...],
    ]:
        # Sample training data
        sample_key, key = jax.random.split(key)
        if isinstance(self.dataset, data.SyntheticData):
            assert self.binarized_classes is None
            test_xs, test_ys, train_xs, train_ys_clean = self._sample_synthetic(
                key=sample_key,
                dataset=self.dataset,
            )
        else:
            assert isinstance(self.dataset, data.DatasetLoader)
            assert (
                self.binarized_classes is not None
            ), "Filter size experiment only supports binary for now"
            test_xs, test_ys, train_xs, train_ys_clean = self._sample_dataset(
                key=sample_key,
                dataset=self.dataset,
            )

        assert test_xs.ndim == 4 and train_xs.ndim == 4
        assert test_xs.shape[1:] == train_xs.shape[1:]
        assert train_ys_clean.shape == (train_xs.shape[0], 1)
        assert test_ys.shape == (test_xs.shape[0], 1)

        if self.crop_size is not None:
            original_height, original_width = train_xs.shape[1:-1]
            crop_height, crop_width = self.crop_size
            assert crop_height <= original_height and crop_width <= original_width
            crop_top = original_height // 2 - crop_height // 2
            crop_left = original_width // 2 - crop_width // 2
            train_xs = train_xs[
                :, crop_top : (crop_top + crop_height), crop_left : (crop_left + crop_width), :
            ]
            test_xs = test_xs[
                :, crop_top : (crop_top + crop_height), crop_left : (crop_left + crop_width), :
            ]

        if self.convert_grayscale and train_xs.shape[-1] == 3:
            # Use same weights as kornia for consistency with torch stuff
            channel_weights = jnp.reshape(
                jnp.array((0.299, 0.587, 0.114), dtype=self.dtype),
                (1, 1, 1, 3),  # NHWC
            )
            train_xs = jnp.sum(train_xs * channel_weights, axis=-1, keepdims=True)
            test_xs = jnp.sum(test_xs * channel_weights, axis=-1, keepdims=True)

        if self.standardize_features:
            # Standardize original images w.r.t. training set statistics
            channels_mean = jnp.mean(train_xs, axis=(0, 1, 2), keepdims=True)
            channels_std = jnp.std(train_xs, axis=(0, 1, 2), keepdims=True)
            assert channels_mean.shape == (
                1,
                1,
                1,
                train_xs.shape[-1],
            ) and channels_std.shape == (1, 1, 1, train_xs.shape[-1])
            train_xs = (train_xs - channels_mean) / channels_std
            test_xs = (test_xs - channels_mean) / channels_std

        self._log.info("Using %d training samples", train_xs.shape[0])
        self._log.info("Using %d test samples", test_xs.shape[0])
        self._metric_logger.log_params(
            {
                "effective_num_train_samples": train_xs.shape[0],
                "effective_num_test_samples": test_xs.shape[0],
            }
        )

        test_loader = util.JAXDataLoader(
            test_xs,
            test_ys,
            batch_size=self.test_batch_size,
            shuffle=False,
        )

        # Apply noise to training labels
        # Do this at the very end to ensure that train xs w/ and w/o noise are exactly the same
        noise_key, key = jax.random.split(key)
        num_noise_labels = int(train_ys_clean.shape[0] * self.noise_fraction)
        assert 0 <= num_noise_labels <= train_xs.shape[0]
        train_noise_mask = jnp.zeros(train_ys_clean.shape[0], dtype=jnp.bool_)
        if num_noise_labels > 0:
            noise_indices_key, noise_labels_key = jax.random.split(noise_key)
            noise_indices = jax.random.permutation(noise_indices_key, train_ys_clean.shape[0])[
                :num_noise_labels
            ]
            train_ys_noise = train_ys_clean.at[noise_indices].set(
                1 - train_ys_clean[noise_indices]
            )

            assert jnp.all(train_ys_clean[noise_indices] != train_ys_noise[noise_indices])
            train_ys = train_ys_noise
            train_noise_mask = train_noise_mask.at[noise_indices].set(True)
        else:
            train_ys = train_ys_clean

        assert train_ys.shape == (train_xs.shape[0], 1)

        train_loader = util.JAXDataLoader(
            train_xs,
            train_ys,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

        train_eval_loader = util.JAXDataLoader(
            train_xs,
            train_ys,
            train_noise_mask,
            batch_size=self.test_batch_size,
            shuffle=False,
        )

        batch_shape = (1,) + train_xs.shape[1:]
        return train_loader, train_eval_loader, test_loader, batch_shape

    def _sample_synthetic(
        self,
        key: jax.random.KeyArray,
        dataset: data.SyntheticData,
    ) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        test_sample_key, train_sample_key = jax.random.split(key)
        test_xs, test_ys = dataset.generate_samples(
            num_samples=self.num_test_samples,
            key=test_sample_key,
            dtype=self.dtype,
        )
        train_xs, train_ys_clean = dataset.generate_samples(
            num_samples=self.num_train_samples,
            key=train_sample_key,
            dtype=self.dtype,
        )
        return test_xs, test_ys, train_xs, train_ys_clean

    def _sample_dataset(
        self,
        key: jax.random.KeyArray,
        dataset: data.DatasetLoader,
    ) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        all_xs, all_ys = dataset.load_dataset()
        all_xs = jnp.array(all_xs.numpy()).astype(self.dtype)
        all_ys = jnp.array(all_ys.numpy()).astype(jnp.int32)

        # JAX uses NHWC but data loader returns NCHW
        all_xs = jnp.transpose(all_xs, axes=(0, 2, 3, 1))

        assert all_xs.ndim == 4 and all_ys.ndim == 1 and all_ys.shape[0] == all_xs.shape[0]

        assert all_ys.min() == 0 and all_ys.max() == dataset.num_classes - 1

        # Stratified split into training and test data
        train_indices = []
        test_indices = []
        if self.binarized_classes is not None:
            # Stratify to have same number of positive and negative samples,
            #  not necessarily same number of samples per original class.
            neg_classes, pos_classes = self.binarized_classes
            assert len(neg_classes) > 0 and len(pos_classes) > 0
            binarized_ys = jnp.zeros_like(all_ys, shape=(all_ys.shape[0], 1))
            num_neg_train_samples_per_class = self.num_train_samples // 2 // len(neg_classes)
            num_pos_train_samples_per_class = self.num_train_samples // 2 // len(pos_classes)
            num_neg_test_samples_per_class = self.num_test_samples // 2 // len(neg_classes)
            num_pos_test_samples_per_class = self.num_test_samples // 2 // len(pos_classes)

            for original_class in range(dataset.num_classes):
                # Always split key to make sure change of classes yields similar results
                shuffle_key, key = jax.random.split(key)
                if original_class not in (neg_classes + pos_classes):
                    continue

                class_indices = jnp.squeeze(jnp.argwhere(all_ys == original_class))
                shuffled_indices = jax.random.permutation(shuffle_key, class_indices.shape[0])

                # Take from opposite ends to ensure one set stays the same if the other one changes
                if original_class in neg_classes:
                    assert (
                        num_neg_test_samples_per_class + num_neg_train_samples_per_class
                        <= class_indices.shape[0]
                    )
                    binarized_ys = binarized_ys.at[all_ys == original_class].set(0)
                    test_indices.append(
                        class_indices[shuffled_indices[:num_neg_test_samples_per_class]]
                    )
                    train_indices.append(
                        class_indices[shuffled_indices[-num_neg_train_samples_per_class:]]
                    )
                elif original_class in pos_classes:
                    assert (
                        num_pos_test_samples_per_class + num_pos_train_samples_per_class
                        <= class_indices.shape[0]
                    )
                    binarized_ys = binarized_ys.at[all_ys == original_class].set(1)
                    test_indices.append(
                        class_indices[shuffled_indices[:num_pos_test_samples_per_class]]
                    )
                    train_indices.append(
                        class_indices[shuffled_indices[-num_pos_train_samples_per_class:]]
                    )
        else:
            raise NotImplementedError()

        train_indices = jnp.concatenate(train_indices)
        test_indices = jnp.concatenate(test_indices)

        test_xs = all_xs[test_indices]
        test_ys = binarized_ys[test_indices]
        train_xs = all_xs[train_indices]
        train_ys_clean = binarized_ys[train_indices]

        return test_xs, test_ys, train_xs, train_ys_clean

    @gin.configurable(denylist=("num_classes", "dtype"))
    def network(
        self,
        num_classes: int,
        cls: typing.Type[flax.linen.Module],
        dtype: typing.Optional[jnp.dtype] = None,
        **network_kwargs,
    ) -> typing.Tuple[flax.linen.Module, typing.Dict[str, typing.AnyStr]]:
        network = cls(num_classes=num_classes, dtype=dtype, **network_kwargs)
        config_dict = {f"network_{key}": str(value) for key, value in network_kwargs.items()}
        config_dict["network"] = cls.__name__
        return network, config_dict

    @gin.configurable(denylist=("steps_per_epoch",))
    def lr_schedule(
        self,
        steps_per_epoch: int,
        builder: typing.Callable[[...], optax._src.base.Schedule],
        per_epoch: bool = False,
        **schedule_kwargs,
    ) -> typing.Tuple[optax._src.base.Schedule, typing.Dict[str, str]]:
        base_schedule = builder(**schedule_kwargs)
        config_dict = {f"schedule_{key}": str(value) for key, value in schedule_kwargs.items()}
        config_dict["schedule"] = builder.__name__
        config_dict["schedule_per_epoch"] = per_epoch
        if per_epoch:
            epochs_per_step = 1.0 / float(steps_per_epoch)

            def schedule(step: chex.Numeric) -> chex.Numeric:
                # Convert from step to epoch (fraction)
                # This allows schedule configuration in terms of epochs instead of steps
                return base_schedule(epochs_per_step * step)

        else:
            schedule = base_schedule

        return schedule, config_dict

    @gin.configurable(denylist=("lr_schedule",))
    def optimizer(
        self,
        lr_schedule: optax._src.base.Schedule,
        builder: typing.Callable[
            [optax._src.alias.ScalarOrSchedule, ...], optax._src.base.GradientTransformation
        ],
        **optimizer_kwargs,
    ) -> typing.Tuple[optax._src.base.GradientTransformation, typing.Dict[str, str]]:
        optimizer = builder(lr_schedule, **optimizer_kwargs)
        config_dict = {f"optimizer_{key}": str(value) for key, value in optimizer_kwargs.items()}
        config_dict["optimizer"] = builder.__name__
        return optimizer, config_dict

    @gin.configurable
    def additional_tags(self, **tags) -> typing.Dict[str, str]:
        # Allows to specify additional tags via gin config
        return {tag: str(value) for tag, value in tags.items()}

    @staticmethod
    def _plot_conv_filters(params: flax.core.FrozenDict):
        filters = params["Conv_0"]["kernel"]
        assert filters.shape[2] == 1
        num_filters = filters.shape[3]
        num_plots_sqrt = math.floor(math.sqrt(num_filters))
        fig, axes = plt.subplots(
            num_plots_sqrt, num_plots_sqrt, figsize=(3 * num_plots_sqrt, 3 * num_plots_sqrt)
        )
        for idx, ax in enumerate(axes.flatten()):
            ax.imshow(filters[:, :, 0, idx], cmap="gray")
        plt.show()
        plt.close(fig)


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
