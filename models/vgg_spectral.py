"""
vgg in pytorch
adapted from:

[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
import functools
from typing import Any, Callable, List, Optional

import torch

import models.layers


cfgs = {
    '5': [64, 'M', 128, 'M', 256, 'M', 512, 'M', ],
    '7': [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M', 512, ],
    '9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, ],
    '11': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, ],
    '14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, ],
    '17': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, ]
}


def make_vgg(cfg: List[Any],
             num_classes: int,
             conv_class: torch.nn.Module,
             activation: torch.nn.Module) -> torch.nn.Sequential:
    layers = []
    num_input_channels = 3
    for l in cfg:
        if l == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [conv_class(num_input_channels, l, kernel_size=3, padding=1)]
        # layers += [torch.nn.BatchNorm2d(l)]
        layers += [activation]
        num_input_channels = l

    layers += [models.layers.RowView(), torch.nn.Linear(512 * 4, num_classes)]
    return torch.nn.Sequential(*layers)


def vgg11(num_classes: int,
          conv_wrapper: Optional[Callable],
          activation_name: str,
          clip_bn: bool) -> torch.nn.Module:
    activation_obj = models.layers.get_activation_obj(activation_name)
    convolution_obj = functools.partial(models.layers.ConvBNBlock,
                                        init_lipschitz=10.0,
                                        conv_wrapper=conv_wrapper,
                                        clip_bn=clip_bn)
    activation = activation_obj()
    vgg = make_vgg(cfgs['11'], num_classes, convolution_obj, activation)
    return vgg
