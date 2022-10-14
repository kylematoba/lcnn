"""
Train models with various flavours of regularization.
"""

import socket
import pprint
import argparse
import time
import random
import os
import logging
import contextlib
import datetime as dt

from functools import partial
from typing import Callable, List, Type

import torch

import path_config
import utils.train
import utils.logging
import models.model_selector
import adversarial_attacks
import evaluations

from models.psoftplus import ParametricSoftplus
from models.layers import ConvBNBlock

from regularized_loss import (CELoss,
                              CurvatureProxyRegularizedLoss,
                              GradNormRegularizedLoss,
                              CurvatureJVPRegularizedLoss,
                              CurvatureProxyAndGradientRegularizedLoss,
                              CURERegularizedLoss)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


attacks_dict = adversarial_attacks.get_attacks_dict()

parser = argparse.ArgumentParser(description='Experiment arguments')
parser.add_argument('-m', '--model-name',
                    type=str,
                    default=None,
                    help="Name of trained model to save, not necessary to specify")
parser.add_argument('--model-arch',
                    default="resnet18", help='What architecture to use?')
parser.add_argument('--regularizer',
                    choices=['gnorm', 'curvature_jvp', 'curvature_proxy', 'curvature_and_gnorm', 'cure'],
                    help='Which regularizer to use? If no option is provided then no regularizer is used')
parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='SGD')
parser.add_argument('--dataset', choices=['cifar10', 'cifar100', "svhn"],
                    default='cifar100',
                    help='Which dataset to use?')
parser.add_argument('--suppress-time', 
                    action='store_true',
                    help="suppress time in logs")
parser.add_argument('--adv_attack',
                    type=str,
                    choices=list(attacks_dict.keys()) + ["None"],
                    help="Adversarial attack name -- configure in adversarial_attacks/get_attacks_dict",
                    default=None)
parser.add_argument("--no_cuda",
                    default=False,
                    action="store_true")
parser.add_argument('--want_cuda',
                    type=bool,
                    default=True,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--run_num', type=int, default=None)

parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--prng_seed", type=int, default=None)
parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
   
parser.add_argument("--mode", type=str, default="", help="Swallow PyCharm args")
parser.add_argument("--port", type=str, default="", help="Swallow PyCharm args")
parser.add_argument("-f", type=str, default="", help="Swallow IPython arg")

args = parser.parse_args()

# https://docs.python.org/3/library/logging.html#levels

suppress_time = args.suppress_time

logging_level = 15
# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
standard_streamhandler = utils.logging.get_standard_streamhandler(not suppress_time)
logger.addHandler(standard_streamhandler)
# End Logging

if args.kwargs is None:
    args.kwargs = {}

cuda_available = torch.cuda.is_available()
want_cuda = not args.no_cuda
use_cuda = want_cuda and cuda_available

if args.prng_seed:
    prng_seed = args.prng_seed
else:
    prng_seed = 0

torch.manual_seed(prng_seed)
random.seed(prng_seed)

# Training hyper-parameters hard-coded for CIFAR10/100
training_args = {
    "momentum": .9,
    'batch_size': 128,
    'lr': 1e-1,
    'weights_l2': 5e-4,
    'gamma': 0.1,
    'softplus_l2': 0.,
    'bn_thresh_l2': 0.,

    # Regularization constants for regularizers
    'reg_beta': 1e-3,
    'reg_gamma': 1e-3,
    'reg_gnorm': 1e-3
}

# If user sets some parameter, use that instead
for tr_arg in training_args:
    if tr_arg in args.kwargs:
        # check the arg's type i.e., int or float or list
        type_val = type(training_args[tr_arg])
        # convert string to arg's 'correct' type
        training_args[tr_arg] = type_val(args.kwargs[tr_arg])

# shorthand for regularizing constants
reg_beta = training_args['reg_beta']
reg_gamma = training_args['reg_gamma']
reg_gnorm = training_args['reg_gnorm']

# Various losses as a dictionary
loss_dict = {
    'gnorm': partial(GradNormRegularizedLoss, reg_constant=reg_gnorm),
    'curvature_jvp': partial(CurvatureJVPRegularizedLoss, reg_constant=reg_beta),
    'curvature_proxy': partial(CurvatureProxyRegularizedLoss, reg_constants=(reg_beta, reg_gamma)),
    'curvature_and_gnorm': partial(CurvatureProxyAndGradientRegularizedLoss, reg_constants=(reg_beta, reg_gamma, reg_gnorm)),
    'cure': partial(CURERegularizedLoss, reg_constant=(reg_gnorm))
}


def _get_named_modules_of_type(model: torch.nn.Module,
                               t: Type[torch.nn.Module]) -> List[torch.nn.Module]:
    return [__ for (_, __) in model.named_modules() if type(__) == t]


def get_optimizer(model: torch.nn.Module,
                  is_smooth_model: bool,
                  training_args: dict) -> torch.optim.Optimizer:
    # Initialize optimizer and scheduler
    if is_smooth_model:
        # Get softplus layers
        psoftplus_paramnames = []
        for (name, layer) in model.named_modules():
            if type(layer) == ParametricSoftplus:
                for p in layer.named_parameters():
                    psoftplus_paramnames.append(name + '.' + p[0])
        bn_paramnames = []
        for (name, layer) in model.named_modules():
            if type(layer) == ConvBNBlock:
                for p in layer.named_parameters():
                    if 'log_lipschitz' in p[0]:
                        bn_paramnames.append(name + '.' + p[0])
        softplus_params = list(map(lambda x: x[1],
                                   list(filter(lambda kv: kv[0] in psoftplus_paramnames, model.named_parameters()))))
        bn_thresh_params = list(map(lambda x: x[1],
                                    list(filter(lambda kv: kv[0] in bn_paramnames, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],
                               list(filter(
                                   lambda kv: (kv[0] not in psoftplus_paramnames) and (kv[0] not in bn_paramnames),
                                   model.named_parameters()))))

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD([
                {'params': base_params, 'weight_decay': training_args['weights_l2']},
                {'params': softplus_params, 'weight_decay': training_args['softplus_l2']},
                {'params': bn_thresh_params, 'weight_decay': training_args['bn_thresh_l2']}],
                lr=training_args["lr"],
                momentum=training_args["momentum"]
            )
        else:
            optimizer = torch.optim.Adam([
                {'params': base_params, 'weight_decay': training_args['weights_l2']},
                {'params': softplus_params, 'weight_decay': training_args['softplus_l2']},
                {'params': bn_thresh_params, 'weight_decay': training_args['bn_thresh_l2']}],
                lr=1e-3
            )
    else:
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=training_args["lr"],
                                        weight_decay=training_args['weights_l2'],
                                        momentum=training_args["momentum"])
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=1e-3,
                                         weight_decay=training_args['weights_l2'])
    return optimizer


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          adversarial_attack: Callable,
          penalty: Callable,
          optimizer: torch.optim.Optimizer,
          is_smooth_model: bool,
          device: torch.device):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device).requires_grad_(), y.to(device)
        if adversarial_attack:
            x = adversarial_attack(model, x)
        optimizer.zero_grad()
        model.zero_grad()

        loss_scalar = penalty.compute_loss(x, y)
        metadata = penalty.metadata

        loss_scalar.backward()
        optimizer.step()

        if 0 == batch_idx % 100:
            log_str = '[{:05d}/{:05d} ({:.0f}%)]'.format(batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader))
            for k, v in metadata.items():
                log_str += '\t' + k + ': {:.4f}'.format(v)
            logger.info(log_str)
            if is_smooth_model:
                psoftplus_layers = _get_named_modules_of_type(model, ParametricSoftplus)
                bn_layers = _get_named_modules_of_type(model, ConvBNBlock)
                for index, module in enumerate(psoftplus_layers):
                    logger.debug("beta{} : {:.4f}".format(index, module.log_beta.exp().data.item()))
                for index, module in enumerate(bn_layers):
                    logger.debug("bn_lipschitz{} : {:.4f}".format(index, module.log_lipschitz.exp().data.item()))


def args_to_modelname(args: argparse.Namespace) -> str:
    if args.model_name is None:
        model_filename = args.model_arch
        if args.regularizer is not None:
            model_filename += '_' + args.regularizer
        if args.dataset is not None:
            model_filename += '_' + args.dataset
        if args.adv_attack is not None:
            model_filename += '_' + args.adv_attack
        if args.prng_seed is not None:
            model_filename += '_' + args.prng_seed
    else:
        model_filename = args.model_name
    return model_filename


if __name__ == "__main__":
    logger.info(f"UTC time {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    logger.info(f"socket.gethostname() = {socket.gethostname()}")

    logger.info(f"model_arch = {args.model_arch}")
    logger.info(f"regularizer = {args.regularizer}")
    adversarial_attack = attacks_dict.get(args.adv_attack, None)
    logger.info(f"adversarial_attack = {adversarial_attack}")
    logger.info(f"dataset = {args.dataset}")

    dtype = None
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info(f"torch.__version__ = {torch.__version__}")
    if use_cuda:
        torch.cuda.manual_seed(prng_seed)
        torch.backends.cudnn.benchmark = True

        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)

        logger.info(f"Running on {current_device_properties}")
        logger.info(f"torch.version.cuda = {torch.version.cuda}")

    paths = path_config.get_paths()
    os.makedirs(paths["saved_models"], exist_ok=True)

    logger.info(pprint.pformat(args.__dict__))
    dataset_name = args.dataset
    logger.info(f"Loading datasets")
    batch_size = training_args['batch_size']
    train_loader, test_loader = utils.train.get_dataloaders(dataset_name, batch_size)
    num_classes = utils.train.get_num_classes(dataset_name)

    num_epochs = args.num_epochs
    model_arch = args.model_arch
    model = models.model_selector.model_architecture[model_arch](num_classes).to(device)

    nb_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"# of parameters {nb_parameters}")

    is_smooth_model = ('smooth' in args.model_arch)
    optimizer = get_optimizer(model, is_smooth_model, training_args)

    loss_fun = loss_dict.get(args.regularizer, CELoss)
    penalty = loss_fun(model)

    milestone_fracs = [0.750, 0.875]
    milestones = [int(_ * num_epochs) for _ in milestone_fracs]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=training_args['gamma'])
    model_filename = args_to_modelname(args)
    model_fullfilename = os.path.join(paths["saved_models"], model_filename)
    logger.info(f"Starting run -- checkpoints will be at {model_fullfilename}")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch: {epoch:03d} / {num_epochs:03d}")
        start = time.time()
        train(model, train_loader, adversarial_attack, penalty, optimizer, is_smooth_model, device)
        logger.info("Time: {:.2f} s".format(time.time() - start))

        total_loss, correct = utils.train.test(model, test_loader, device, dtype)
        average_loss = total_loss / len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(average_loss,
                                                                                           correct,
                                                                                           len(test_loader.dataset),
                                                                                           acc))
        scheduler.step()
        if use_cuda:
            memory_summary = torch.cuda.memory_summary()
            logger.debug(memory_summary)
        # Save the current model
        if epoch < num_epochs:
            epoch_fullfilename = f"{model_fullfilename}_{epoch}.pt"
        else:
            epoch_fullfilename = f"{model_fullfilename}.pt"

        logger.debug(f"Saving to {epoch_fullfilename}")
        torch.save(model.cpu().state_dict(), epoch_fullfilename)

        # delete previously stored model
        if epoch > 1:
            with contextlib.suppress(FileNotFoundError):
                os.remove(f"{model_fullfilename}_{epoch - 1}.pt")
        model.to(device)

    logger.info(f"{model_fullfilename}")

    # for key, value in model.cpu().state_dict().items():
    #     # print(f"key = {key} -> value = {value[:4]}")
    #     logger.info(f"key = {key}")
    #     if 0 == value.ndim:
    #         logger.info(f"-> value = {value}")
    #     elif 1 == value.ndim:
    #         logger.info(f"-> value = {value[:3]}")
    #     elif 2 == value.ndim:
    #         logger.info(f"-> value = {value[:3, :3]}")
    #     elif 3 == value.ndim:
    #         logger.info(f"-> value = {value[:3, :3, 0]}")
    #     elif 4 == value.ndim:
    #         logger.info(f"-> value = {value[:3, :3, 0, 0]}")
    #     else:
    #         raise ValueError(f"Not prepared to handle value.ndim > 4")

    max_batches = 10
    model = model.to(device)
    _, test_correct = utils.train.test(model, test_loader, device, dtype)
    c, h, g = evaluations.evaluate_curvature(model, test_loader, max_batches, batch_size, device)

    acc = 100. * test_correct / len(test_loader.dataset)
    logger.info('Model, Avg Curvature, Std Curvature, Avg Hessian Sp.Norm, Std Hessian Sp.Norm, Avg Gradnorm, Std Gradnorm, accuracy')
    to_log = f"{model_filename}, {c[0]:8.3f}, {c[1]:8.3f}, {h[0]:8.3f}, {h[1]:8.3f}, {g[0]:8.3f}, {g[1]:8.3f}, {acc:.3f}"
    logger.info(to_log)
    logger.info(f"[] Fitting done. See {model_fullfilename}.pt")
