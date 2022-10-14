from functools import partial

import torch.nn.utils.parametrizations

import models.resnet_spectral
import models.conv_spectral_norm
import models.vgg_spectral


def union_dicts(x: dict, y: dict) -> dict:
    return {**x, **y}


baselines = {
    'resnet18': dict(func=models.resnet_spectral.resnet18,
                     conv_wrapper=None,
                     activation_name='softplus',
                     clip_bn=False),
    'resnet34': dict(func=models.resnet_spectral.resnet34,
                     conv_wrapper=None,
                     activation_name='softplus',
                     clip_bn=False),
    "vgg11": dict(func=models.vgg_spectral.vgg11,
                  conv_wrapper=None,
                  activation_name='softplus',
                  clip_bn=False),
}

smooths = {
    # use parametric softplus and spectral norm for "smooth" models
    'resnet18smooth': dict(func=models.resnet_spectral.resnet18,
                           conv_wrapper=torch.nn.utils.parametrizations.spectral_norm,
                           activation_name='parametric_softplus',
                           clip_bn=True),
    'resnet34smooth': dict(func=models.resnet_spectral.resnet34,
                           conv_wrapper=torch.nn.utils.parametrizations.spectral_norm,
                           activation_name='parametric_softplus',
                           clip_bn=True),
    'vgg11smooth': dict(func=models.vgg_spectral.vgg11,
                        conv_wrapper=torch.nn.utils.parametrizations.spectral_norm,
                        activation_name='parametric_softplus',
                        clip_bn=True),
}

smoothconvs = {
    'resnet18smoothconv': dict(func=models.resnet_spectral.resnet18,
                               conv_wrapper=models.conv_spectral_norm.convspectralnorm_wrapper,
                               activation_name='parametric_softplus',
                               clip_bn=True),
    'resnet34smoothconv': dict(func=models.resnet_spectral.resnet34,
                               conv_wrapper=models.conv_spectral_norm.convspectralnorm_wrapper,
                               activation_name='parametric_softplus',
                               clip_bn=True),
    "vgg11smoothconv": dict(func=models.vgg_spectral.vgg11,
                            conv_wrapper=models.conv_spectral_norm.convspectralnorm_wrapper,
                            activation_name='parametric_softplus',
                            clip_bn=True),
}

approxs = {
    'resnet18smoothconvapprox': dict(func=models.resnet_spectral.resnet18,
                                     conv_wrapper=None,
                                     activation_name='parametric_softplus',
                                     clip_bn=True),
    "vgg11smoothconvapprox": dict(func=models.vgg_spectral.vgg11,
                            conv_wrapper=None,
                            activation_name='parametric_softplus',
                            clip_bn=True),
}

ablations = {
    # baselines and ablations
}

architecture_config = dict()
architecture_config = union_dicts(architecture_config, baselines)
architecture_config = union_dicts(architecture_config, ablations)
architecture_config = union_dicts(architecture_config, smooths)
architecture_config = union_dicts(architecture_config, smoothconvs)
architecture_config = union_dicts(architecture_config, approxs)


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def config_to_model(config: dict) -> torch.nn.Module:
    model = partial(config["func"], **without_keys(config, ["func"]))
    return model


model_architecture = {
    k: config_to_model(v) for k, v in architecture_config.items()
}


if __name__ == "__main__":
    model_name = "resnet34"

    config = architecture_config[model_name]
    model = config_to_model(config)
