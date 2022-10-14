"""
Adapted from:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import functools
from typing import Callable, List, Optional, Tuple

import torch

from models.conv_spectral_norm import convspectralnorm_wrapper
import models.layers


def get_basic_block(in_channels: int,
                    out_channels: int,
                    stride: int,
                    activation_obj: torch.nn.Module,
                    conv_obj: torch.nn.Module) -> torch.nn.Module:
    conv1 = conv_obj(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
    conv2 = conv_obj(out_channels,
                     out_channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)
    conv_shortcut = torch.nn.Identity()

    if stride != 1 or in_channels != out_channels:
        conv_shortcut = conv_obj(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=stride,
                                 padding=0,
                                 bias=False)
    layers = [models.layers.ChannelRepeatOnce(),
              models.layers.ApplyOverHalves(conv1, conv_shortcut),
              models.layers.ApplyOverHalves(activation_obj(), torch.nn.Identity()),
              models.layers.ApplyOverHalves(conv2, torch.nn.Identity()),
              models.layers.SumHalves(),
              activation_obj()]
    model = torch.nn.Sequential(*layers)
    return model


def _make_layer(in_channels: int,
                out_channels: int,
                num_blocks: int,
                stride: int,
                activation_obj,
                conv_obj) -> Tuple[torch.nn.Module, int]:
    # we have num_block blocks per layer, the first block
    # could be 1 or 2, other blocks would always be 1
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        basic_block = get_basic_block(in_channels,
                                      out_channels,
                                      stride,
                                      activation_obj,
                                      conv_obj)
        layers.append(basic_block)
        in_channels = out_channels
    return torch.nn.Sequential(*layers), in_channels


def _get_resnet(num_block: List[int],
                num_classes: int,
                activation_obj,
                convolution_obj) -> torch.nn.Module:
    in_channels = 64
    conv1 = torch.nn.Sequential(
        convolution_obj(3, in_channels, kernel_size=3, padding=1, bias=False, stride=1),
        activation_obj()
    )
    conv2_x, in_channels = _make_layer(in_channels, 64, num_block[0], 1,
                                       activation_obj,
                                       convolution_obj)
    conv3_x, in_channels = _make_layer(in_channels, 128, num_block[1], 2,
                                       activation_obj,
                                       convolution_obj)
    conv4_x, in_channels = _make_layer(in_channels, 256, num_block[2], 2,
                                       activation_obj,
                                       convolution_obj)
    conv5_x, in_channels = _make_layer(in_channels, 512, num_block[3], 2,
                                       activation_obj,
                                       convolution_obj)
    avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    fc = torch.nn.Linear(512, num_classes)
    fc = torch.nn.utils.parametrizations.spectral_norm(fc)
    view = models.layers.RowView()

    layers = [conv1,
              conv2_x,
              conv3_x,
              conv4_x,
              conv5_x,
              avg_pool,
              view,
              fc]
    resnet = torch.nn.Sequential(*layers)
    return resnet


def resnet18(num_classes: int,
             conv_wrapper: Optional[Callable],
             activation_name: str,
             clip_bn: bool) -> torch.nn.Module:
    num_block = [2, 2, 2, 2]
    activation_obj = models.layers.get_activation_obj(activation_name)
    convolution_obj = functools.partial(models.layers.ConvBNBlock,
                                        init_lipschitz=10.0,
                                        conv_wrapper=conv_wrapper,
                                        clip_bn=clip_bn)
    resnet = _get_resnet(num_block,
                         num_classes,
                         activation_obj,
                         convolution_obj)
    if conv_wrapper is convspectralnorm_wrapper:
        # Initialize conv spectral norm layers
        x = torch.randn((1, 3, 32, 32))
        resnet(x)
    return resnet


def resnet34(num_classes: int,
             conv_wrapper: Callable,
             activation_name: str,
             clip_bn: bool) -> torch.nn.Module:
    activation_obj = models.layers.get_activation_obj(activation_name)
    convolution_obj = functools.partial(models.layers.ConvBNBlock,
                                        init_lipschitz=10.0,
                                        conv_wrapper=conv_wrapper,
                                        clip_bn=clip_bn)
    num_block = [3, 4, 6, 3]
    resnet = _get_resnet(num_block,
                         num_classes,
                         activation_obj,
                         convolution_obj)

    if conv_wrapper is convspectralnorm_wrapper:
        # Initialize conv spectral norm layers
        x = torch.randn((1, 3, 32, 32))
        resnet(x)
    return resnet


def flatten_sequential_of_sequential(model: torch.nn.Module) -> torch.nn.Module:
    assert type(model) == torch.nn.Sequential
    flattened_layers = []
    for layer in model:
        if type(layer) == torch.nn.Sequential:
            to_add = list(flatten_sequential_of_sequential(layer))
        else:
            to_add = [layer]
        flattened_layers += to_add
    flattened_model = torch.nn.Sequential(*flattened_layers)
    return flattened_model


if __name__ == "__main__":
    num_classes = 100

    conv_wrapper = None
    activation_name = 'softplus'
    test_model = resnet18(num_classes, conv_wrapper, activation_name)

    print(test_model)
