"""

Very heavily based on
E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, and W. Yin.
"Plug-and-Play Methods Provably Converge with Properly Trained Denoisers."
ICML, 2019.

(https://github.com/uclaopt/Provable_Plug_and_Play/blob/master/training/model/conv_sn_chen.py)
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


def _normalize(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    norm = max(float(torch.sqrt(torch.sum(x.pow(2)))), eps)
    return torch.div(x, norm, out=x)


class ConvSpectralNorm(torch.nn.Module):
    def __init__(self, 
                weight: torch.Tensor,
                stride: Union[int, tuple] = 1,
                dilation: Union[int, tuple] = 1,
                padding: Union[int, tuple] = 1,
                im_size: int = 10,
                n_power_iterations: int = 1,
                eps: float = 1e-12):
        super().__init__()
        num_iterations_initial = 15

        self.im_size = im_size
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        self.padding = padding[0]
        self.stride = stride[0]
        self.dilation = dilation[0]

        # Left singular vectors
        u = _normalize(weight.new_empty((1, weight.size(0), self.im_size, self.im_size)).normal_(0, 1), eps=self.eps)
        v = _normalize(weight.new_empty((1, weight.size(1), self.im_size, self.im_size)).normal_(0, 1), eps=self.eps)

        self.register_buffer('_u', u)
        self.register_buffer('_v', v)

        self._power_method(weight, num_iterations_initial)

    @torch.autograd.no_grad()
    def _power_method(self, weight: torch.Tensor, num_iterations: int) -> None:
        for _ in range(num_iterations):
            # Spectral norm of weight equals to `u^T * W * v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            # print('Before:', self._v.size(), self._u.size())
            self._v = _normalize(torch.nn.functional.conv_transpose2d(self._u,weight, 
                                padding=self.padding, 
                                stride=self.stride,
                                dilation=self.dilation),
                                eps=self.eps)
            self._u = _normalize(torch.nn.functional.conv2d(self._v, weight, 
                                padding=self.padding,
                                stride=self.stride,
                                dilation=self.dilation), 
                                eps=self.eps)
            # print('After:', self._v.size(), self._u.size())

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(weight, self.n_power_iterations)
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        spectral_norm = torch.sum(u * torch.nn.functional.conv2d(v, weight, 
                                                        padding=self.padding,
                                                        stride=self.stride,
                                                        dilation=self.dilation))
        if spectral_norm < self.eps:
            return weight 
        else:
            return weight / spectral_norm


def convspectralnorm_wrapper(module: torch.nn.Module,
                             im_size: int=10,
                             n_power_iterations: int=1):
    return P.register_parametrization(module, "weight", 
                                ConvSpectralNorm(weight=module.weight, 
                                                 stride=module.stride,
                                                 dilation=module.dilation,
                                                 padding=module.padding,
                                                 im_size=im_size,
                                                 n_power_iterations=n_power_iterations))


if __name__ == "__main__":
    size_in = (16, 16)
    in_channels = 19
    out_channels = 37
    conv1 = nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=2)
    conv1 = convspectralnorm_wrapper(conv1)
    print(conv1)
    x = torch.randn(size=(1, in_channels, 40, 40))
    y = conv1(x)
