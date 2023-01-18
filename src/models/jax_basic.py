import typing

import flax
import flax.traverse_util
import gin
import jax.numpy as jnp


@gin.configurable
class FilterSizeCNN(flax.linen.Module):
    filter_size: int
    width: int
    dense_width: int
    num_classes: int
    pre_pool_widths: typing.Tuple[int, ...] = ()
    dtype: typing.Optional[jnp.dtype] = None

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Conv(
            features=self.width,
            kernel_size=(self.filter_size, self.filter_size),
            strides=1,
            padding="VALID",
            use_bias=True,
            dtype=self.dtype,
            kernel_init=flax.linen.initializers.variance_scaling(
                scale=2.0,  # for VARIANCE, not stddev
                mode="fan_out",
                distribution="normal",
            ),
        )(x)
        for pre_pool_width in self.pre_pool_widths:
            x = flax.linen.relu(x)
            x = flax.linen.Conv(
                features=pre_pool_width,
                kernel_size=(1, 1),
                strides=1,
                padding="VALID",
                use_bias=True,
                dtype=self.dtype,
                kernel_init=flax.linen.initializers.variance_scaling(
                    scale=2.0,  # for VARIANCE, not stddev
                    mode="fan_out",
                    distribution="normal",
                ),
            )(x)
        x = jnp.max(x, axis=(1, 2))  # global max pooling
        x = flax.linen.relu(x)  # relu after pooling to save resources

        x = jnp.reshape(x, (x.shape[0], -1))  # flatten

        x = flax.linen.Dense(features=self.dense_width, use_bias=True, dtype=self.dtype)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes, use_bias=True, dtype=self.dtype)(x)

        return x
