import abc
import logging
import math
import os
import typing

import cairo
import gin
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision
import torchvision.datasets.utils

import util


class DatasetLoader(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_dataset(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # returns xs, ys
        pass

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sample_shape(self) -> torch.Size:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def config(self) -> typing.Dict[str, str]:
        return dict()


class SyntheticData(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_samples(
        self,
        num_samples: int,
        key: jax.random.KeyArray,
        dtype: jnp.dtype = jnp.float32,
    ) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
        # returns xs, ys
        pass

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sample_shape(self) -> typing.Tuple[int, ...]:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def config(self) -> typing.Dict[str, str]:
        pass


@gin.configurable
class EuroSATLoader(DatasetLoader):
    def load_dataset(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        raw_dataset = torchvision.datasets.EuroSAT(
            root=str(util.get_data_dir()),
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        raw_xs = []
        raw_ys = []
        for x, y in raw_dataset:
            raw_xs.append(x)
            raw_ys.append(y)
        xs = torch.stack(raw_xs, dim=0)
        ys = torch.tensor(raw_ys)
        return xs, ys

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def sample_shape(self) -> torch.Size:
        return torch.Size((3, 64, 64))

    @property
    def name(self) -> str:
        return "EuroSAT"


@gin.configurable
class ShapeDataset(SyntheticData):
    def __init__(
        self,
        image_size: int = 32,
        shape_size: float = 5,
        min_shape_size: typing.Optional[float] = None,
        num_shapes_per_sample: int = 10,
        force_inside: bool = True,
        use_squares: bool = False,
        use_background: bool = False,
        cache_dir: typing.Optional[str] = None,
    ):
        self._image_size = image_size
        self._shape_size = shape_size
        self._min_shape_size = min_shape_size
        self._num_shapes_per_sample = num_shapes_per_sample
        self._force_inside = force_inside
        self._use_squares = use_squares
        self._use_background = use_background
        self._cache_dir = (
            os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir is not None else None
        )
        self._log = logging.getLogger(self.__class__.__name__)

    def generate_samples(
        self,
        num_samples: int,
        key: jax.random.KeyArray,
        dtype: jnp.dtype = jnp.float32,
    ) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
        num_negative_samples = num_samples // 2
        num_positive_samples = num_samples - num_negative_samples
        ys = jnp.zeros((num_samples, 1), dtype=jnp.int32)
        ys = ys.at[:num_positive_samples].set(1)

        if self._cache_dir is not None:
            os.makedirs(self._cache_dir, exist_ok=True)
            cache_path = os.path.join(
                self._cache_dir,
                f"{self._image_size}_{self._shape_size}-{self._min_shape_size}_{self._num_shapes_per_sample}_{'fi' if self._force_inside else 'nfi'}_{'sq' if self._use_squares else 'circ'}_{'bg' if self._use_background else 'nobg'}_{key[0]}_{key[1]}_{num_samples}.npy",
            )
            if os.path.exists(cache_path):
                self._log.debug("Loading cached images from %s", cache_path)
                xs_raw = np.load(cache_path)
            else:
                xs_raw = self._generate_samples_internal(labels=ys, key=key)
                self._log.debug("Saving images to %s", cache_path)
                np.save(cache_path, xs_raw)

                # Uncomment the following block to plot some samples:
                # import matplotlib.pyplot as plt
                #
                # num_img_per_class = 5
                # fig, axes = plt.subplots(num_img_per_class, 2,
                #                          figsize=(10, 5 * num_img_per_class))
                # for idx in range(num_img_per_class):
                #     axes[idx, 0].imshow(xs_raw[idx, :, :, 0])
                #     axes[idx, 1].imshow(xs_raw[-(idx + 1), :, :, 0])
                # plt.show()
                # plt.close(fig)

        else:
            xs_raw = self._generate_samples_internal(labels=ys, key=key)

        xs = jnp.array(xs_raw, dtype=dtype)
        return xs, ys

    def _generate_samples_internal(
        self,
        labels: jnp.ndarray,
        key: jax.random.KeyArray,
    ) -> np.ndarray:
        num_samples = labels.shape[0]
        xs_raw = np.zeros((num_samples, self._image_size, self._image_size, 1), dtype=np.float32)
        for idx in range(num_samples):
            current_class = int(labels[idx])
            current_key, key = jax.random.split(key)
            current_sample = self._generate_single_sample(
                label=current_class,
                key=current_key,
            )
            xs_raw[idx, :, :, 0] = current_sample
            del current_sample
        return xs_raw

    def _generate_single_sample(
        self,
        label: int,
        key: jax.random.KeyArray,
    ) -> np.ndarray:
        full_image_size = int(2 ** math.ceil(math.log2(self._image_size)))

        background_key, key = jax.random.split(key)
        if self._use_background:
            # Sample white noise background
            background = jax.random.randint(
                key, (full_image_size, full_image_size), 0, 256, dtype=jnp.uint8
            )
            background = np.array(background)
            # noinspection PyTypeChecker
            surface = cairo.ImageSurface.create_for_data(
                background, cairo.FORMAT_A8, full_image_size, full_image_size
            )
        else:
            surface = cairo.ImageSurface(cairo.FORMAT_A8, full_image_size, full_image_size)
        ctx = cairo.Context(surface)

        ctx.set_line_width(0.5)

        for _ in range(self._num_shapes_per_sample):
            pos_key, angle_key, size_key, key = jax.random.split(key, 4)
            if self._min_shape_size is not None:
                shape_size = float(
                    jax.random.uniform(
                        size_key, minval=self._min_shape_size, maxval=self._shape_size
                    )
                )
            else:
                shape_size = self._shape_size

            border_distance = shape_size / 2.0 if self._force_inside else 0.0
            x, y = jax.random.uniform(
                pos_key,
                shape=(2,),
                minval=border_distance,
                maxval=self._image_size - border_distance,
            )
            angle = jax.random.uniform(angle_key, minval=0.0, maxval=2.0 * math.pi)

            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(-angle)
            ctx.scale(shape_size / 2.0, shape_size / 2.0)

            if label == 0:
                if self._use_squares:
                    # Square
                    ctx.move_to(-1.0, -1.0)
                    ctx.line_to(1.0, -1.0)
                    ctx.line_to(1.0, 1.0)
                    ctx.line_to(-1.0, 1.0)
                    ctx.close_path()
                    ctx.stroke()
                else:
                    # Circle
                    ctx.arc(0.0, 0.0, 1.0, 0.0, 2.0 * math.pi)
                    ctx.stroke()
            else:
                # X
                assert label == 1
                ctx.move_to(-1.0, -1.0)
                ctx.line_to(1.0, 1.0)
                ctx.stroke()

                ctx.move_to(-1.0, 1.0)
                ctx.line_to(1.0, -1.0)
                ctx.stroke()

            ctx.restore()

        buf = surface.get_data()
        data = np.ndarray(shape=(full_image_size, full_image_size), dtype=np.uint8, buffer=buf)
        data = data[: self._image_size, : self._image_size]
        data = data.astype(np.float32) / 255.0
        return data

    @property
    def config(self) -> typing.Dict[str, str]:
        return {
            "image_size": str(self._image_size),
            "shape_size": str(self._shape_size),
            "min_shape_size": str(self._min_shape_size),
            "num_shapes_per_sample": str(self._num_shapes_per_sample),
            "force_inside": str(self._force_inside),
            "use_squares": str(self._use_squares),
            "use_background": str(self._use_background),
            "use_cache": str(self._cache_dir is not None),
        }

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def sample_shape(self) -> typing.Tuple[int, ...]:
        return self._image_size, self._image_size, 1

    @property
    def name(self) -> str:
        return "Shapes"
