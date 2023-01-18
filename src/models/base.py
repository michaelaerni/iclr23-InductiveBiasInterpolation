import abc
import typing

import torch


class NetworkBuilder(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(
        self,
        sample_shape: torch.Size,
        num_classes: int,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        pass

    @property
    @abc.abstractmethod
    def config_dict(self) -> typing.Dict[str, typing.Any]:
        pass
