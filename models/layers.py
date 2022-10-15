import math
import functools
from typing import Callable, Optional, Tuple, Union

import torch
import models.conv_spectral_norm
import models.psoftplus

Conv2dArg = Union[int, Tuple[int, int]]


class ApplyOverHalves(torch.nn.Module):
    def __init__(self, f1: Callable, f2: Callable):
        super().__init__()
        self.f1 = f1
        self.f2 = f2

    def __call__(self, x: torch.Tensor):
        assert 4 == x.ndim
        assert 0 == x.shape[1] % 2
        dim = int(x.shape[1] / 2)
        x = torch.cat((self.f1(x[:, :dim, :, :]),
                       self.f2(x[:, dim:, :, :])), axis=1)
        return x


class ChannelRepeatOnce(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        assert 4 == x.ndim
        return x.repeat(1, 2, 1, 1) / math.sqrt(2)


class SumHalves(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor):
        assert 4 == x.ndim
        assert 0 == x.shape[1] % 2
        dim = int(x.shape[1] / 2)
        x = x[:, :dim, :, :] + x[:, dim:, :, :]
        return 0.5 * x


class RowView(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ConvBNBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding: int = 0,
                 bias: bool = True,
                 stride: int = 1,
                 kernel_size: int = 3,
                 init_lipschitz: float = math.inf,
                 conv_wrapper: Optional[Callable]=None,
                 clip_bn: Optional[bool]=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        # Initialize wrapper if it is *not* conv spectral norm
        if (conv_wrapper is not None) and \
           (conv_wrapper is not models.conv_spectral_norm.convspectralnorm_wrapper):
            self.conv = conv_wrapper(self.conv)

        self.bn = torch.nn.BatchNorm2d(out_channels, affine=False)
        self.log_lipschitz = torch.nn.Parameter(torch.tensor(init_lipschitz).log())
        self.conv_wrapper = conv_wrapper
        self.init_convspectralnorm = True
        if clip_bn is None:
            clip_bn = self.conv_wrapper is not None
        # self.clip_bn = self.conv_wrapper is not None
        self.clip_bn = clip_bn

    def forward(self, x):
        if self.conv_wrapper is not None:
            # initialize conv spectral norm taking into account the size of the input
            if self.init_convspectralnorm is True:
                if self.conv_wrapper is models.conv_spectral_norm.convspectralnorm_wrapper:
                    self.conv = self.conv_wrapper(self.conv, im_size=x.size(2))
                self.init_convspectralnorm = False
        x = self.conv(x)
        if self.clip_bn:
            scale = torch.min((self.bn.running_var + 1e-5) ** .5)
            one_lipschitz_part = self.bn(x) * scale
            x = one_lipschitz_part * torch.minimum(1 / scale, self.log_lipschitz.exp())
        else:
            x = self.bn(x)
        return x


PartialModules = [ApplyOverHalves]

LinearModules = [ChannelRepeatOnce,
                 SumHalves,
                 RowView,
                 ConvBNBlock]


def get_activation_obj(activation_name: str) -> torch.nn.Module:
    if activation_name == 'parametric_softplus':
        activation_obj = models.psoftplus.ParametricSoftplus
    elif activation_name == 'softplus':
        activation_obj = functools.partial(torch.nn.Softplus, beta=1e3)
    elif activation_name == 'low_softplus':
        activation_obj = functools.partial(torch.nn.Softplus, beta=10)
    else:
        raise ValueError(f"unknown activation_name {activation_name}")
    return activation_obj
