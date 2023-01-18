import typing

import gin
import torch
import torch.nn as nn

from . import base


class WideBlock(nn.Module):
    def __init__(
        self,
        features_in: int,
        features: int,
        stride: int,
        expand_features: bool,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ):
        super(WideBlock, self).__init__()

        self.norm_in = nn.BatchNorm2d(features_in, device=device, dtype=dtype)
        self.conv_in = nn.Conv2d(
            features_in,
            features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.norm_out = nn.BatchNorm2d(features, device=device, dtype=dtype)
        self.conv_out = nn.Conv2d(
            features,
            features,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            device=device,
            dtype=dtype,
        )

        if stride != 1 or expand_features:
            self.identity = nn.Conv2d(
                features_in,
                features,
                kernel_size=(1, 1),
                stride=stride,
                device=device,
                dtype=dtype,
            )
        else:
            self.identity = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm_in(inputs))
        x = self.conv_in(x)

        x = torch.relu(self.norm_out(x))
        x = self.conv_out(x)

        x += self.identity(inputs)

        return x


class WideResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        widen_factor: int,
        num_classes: int,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0 and depth > 4
        assert widen_factor > 0
        num_blocks = (depth - 4) // 6
        features_per_block = (
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        )

        features_in = features_per_block[0]
        self.conv_in = nn.Conv2d(
            in_channels,
            features_in,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=device,
            dtype=dtype,
        )

        self.group_1, features_in = self._wide_layer(
            features_in,
            features_per_block[1],
            stride=1,
            num_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )
        self.group_2, features_in = self._wide_layer(
            features_in,
            features_per_block[2],
            stride=2,
            num_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )
        self.group_3, features_in = self._wide_layer(
            features_in,
            features_per_block[3],
            stride=2,
            num_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )

        self.norm_out = nn.BatchNorm2d(features_in, device=device, dtype=dtype)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.dense = nn.Linear(features_in, num_classes, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        x = self.group_1(x)
        x = self.group_2(x)
        x = self.group_3(x)

        x = torch.relu(self.norm_out(x))

        # Pool and flatten
        x = self.flatten(self.pool(x))

        x = self.dense(x)

        return x

    @staticmethod
    def _wide_layer(
        features_in: int,
        features: int,
        num_blocks: int,
        stride: int,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ) -> typing.Tuple[nn.Module, int]:
        blocks = []
        for block_idx in range(num_blocks):
            block = WideBlock(
                features_in=features_in,
                features=features,
                stride=stride if block_idx == 0 else 1,
                expand_features=(features_in != features),
                device=device,
                dtype=dtype,
            )
            blocks.append(block)
            features_in = features

        return nn.Sequential(*blocks), features_in


@gin.configurable
class WideResNetBuilder(base.NetworkBuilder):
    def __init__(self, depth: int, widen_factor: int):
        self._depth = depth
        self._widen_factor = widen_factor

    def build(
        self,
        sample_shape: torch.Size,
        num_classes: int,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        num_channels, _, _ = sample_shape

        return WideResNet(
            in_channels=num_channels,
            depth=self._depth,
            widen_factor=self._widen_factor,
            num_classes=num_classes,
            device=device,
            dtype=dtype,
        )

    @property
    def config_dict(self) -> typing.Dict[str, typing.Any]:
        return {"depth": self._depth, "widen_factor": self._widen_factor}
