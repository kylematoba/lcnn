"Estimate model curvature using the power method"

import os
import logging
from typing import Tuple
import argparse
import socket
import datetime as dt
import random

import torch
import torch.nn.functional as F

import utils.logging
import utils.path_config as path_config
import models.model_selector
import utils.train

# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(utils.logging.get_standard_streamhandler())


def curvature_hessian_estimator(model: torch.nn.Module,
                        image: torch.Tensor,
                        target: torch.Tensor,
                        num_power_iter: int=20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    model.eval()
    u = torch.randn_like(image)
    u /= torch.norm(u, p=2, dim=(1, 2, 3), keepdim=True)

    with torch.enable_grad():
        image = image.requires_grad_()
        out = model(image)
        y = F.log_softmax(out, 1)
        output = F.nll_loss(y, target, reduction='none')
        model.zero_grad()
        # Gradients w.r.t. input
        gradients = torch.autograd.grad(outputs=output.sum(),
                                        inputs=image, create_graph=True)[0]
        gnorm = torch.norm(gradients, p=2, dim=(1, 2, 3))
        assert not gradients.isnan().any()

        # Power method to find singular value of Hessian
        for _ in range(num_power_iter):
            grad_vector_prod = (gradients * u.detach_()).sum()
            hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image, retain_graph=True)[0]
            assert not hessian_vector_prod.isnan().any()

            hvp_norm = torch.norm(hessian_vector_prod, p=2, dim=(1, 2, 3), keepdim=True)
            u = hessian_vector_prod.div(hvp_norm + 1e-6) #1e-6 for numerical stability

        grad_vector_prod = (gradients * u.detach_()).sum()
        hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image)[0]
        hessian_singular_value = (hessian_vector_prod * u.detach_()).sum((1, 2, 3))
    
    # curvature = hessian_singular_value / (grad_norm + epsilon) by definition
    curvatures = hessian_singular_value.abs().div(gnorm + 1e-6)
    hess = hessian_singular_value.abs()
    grad = gnorm
    
    return curvatures, hess, grad


def measure_curvature(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      data_fraction: float=0.1,
                      batch_size: int=64,
                      num_power_iter: int=20,
                      device: torch.device='cpu') -> Tuple[tuple, tuple, tuple]:

    """
    Compute curvature, hessian norm and gradient norm of a subset of the data given by the dataloader.
    These values are computed using the power method, which requires setting the number of power iterations (num_power_iter).
    """

    model.eval()
    datasize = int(data_fraction * len(dataloader.dataset))
    max_batches = int(datasize / batch_size)
    curvature_agg = torch.empty(size=(datasize,))
    grad_agg = torch.empty(size=(datasize,))
    hess_agg = torch.empty(size=(datasize,))

    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device).requires_grad_(), target.to(device)
        with torch.no_grad():
            curvatures, hess, grad = curvature_hessian_estimator(model, data, target, num_power_iter=num_power_iter)
        curvature_agg[idx * batch_size:(idx + 1) * batch_size] = curvatures.detach()
        hess_agg[idx * batch_size:(idx + 1) * batch_size] = hess.detach()
        grad_agg[idx * batch_size:(idx + 1) * batch_size] = grad.detach()

        avg_curvature, std_curvature = curvature_agg.mean().item(), curvature_agg.std().item()
        avg_hessian, std_hessian = hess_agg.mean().item(), hess_agg.std().item()
        avg_grad, std_grad = grad_agg.mean().item(), grad_agg.std().item()

        if idx == (max_batches - 1):
            logger.info('Average Curvature: {:.6f} +/- {:.2f} '.format(avg_curvature, std_curvature))
            logger.info('Average Hessian Spectral Norm: {:.6f} +/- {:.2f} '.format(avg_hessian, std_hessian))
            logger.info('Average Gradient Norm: {:.6f} +/- {:.2f}'.format(avg_grad, std_grad))
            return


def main():

    parser = argparse.ArgumentParser(description='Experiment arguments')

    parser.add_argument('--model-arch',
                        default="resnet18", 
                        help='What architecture to use?')

    parser.add_argument('--model-filename',
                        type=str, 
                        help='Full filename with path')
    
    parser.add_argument('--dataset', 
                        choices=['cifar10', 'cifar100', "svhn"],
                        default='cifar100',
                        help='Which dataset to use?')

    parser.add_argument("--data-fraction", 
                        type=float, 
                        default=0.1,
                        help="Fraction of data to use for curvature estimation")
    
    parser.add_argument("--batch-size", 
                        type=int, 
                        default=64)

    parser.add_argument('--num-power-iter',
                        type=int,
                        default=10,
                        help="# power iterations for power method")

    parser.add_argument("--prng_seed", 
                        type=int, 
                        default=1729)

    args = parser.parse_args()

    # Show user some information about current job
    logger.info(f"UTC time {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Host: {socket.gethostname()}")

    logger.info("\n----------------------------")
    logger.info("    Argparse arguments")
    logger.info("----------------------------")
    # print all argparse'd args
    for arg in vars(args):
        logger.info(f"{arg} \t {getattr(args, arg)}")
    
    logger.info("----------------------------\n")

    return args

def get_model_and_datasets(args):
     # Random seeds
    prng_seed = args.prng_seed
    torch.manual_seed(prng_seed)
    random.seed(prng_seed)

    # Device selection   
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")    
    if cuda_available:
        torch.cuda.manual_seed(prng_seed)
        torch.backends.cudnn.benchmark = True
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        logger.info(f"Running on {current_device_properties}")
        logger.info(f"torch.version.cuda = {torch.version.cuda}")

    paths = path_config.get_paths()

    # Load Datasets
    logger.info(f"Loading datasets")
    os.makedirs(paths[args.dataset], exist_ok=True)
    train_loader, test_loader = utils.train.get_dataloaders(args.dataset, args.batch_size)
    num_classes = utils.train.get_num_classes(args.dataset)

    #Load Model
    model = models.model_selector.model_architecture[args.model_arch](num_classes)
    model.load_state_dict(torch.load(args.model_filename), strict=True)

    return model, train_loader, test_loader, device

if __name__ == "__main__":
    args = main()
    model, train_loader, test_loader, device = get_model_and_datasets(args)

    logger.info("\nEstimating curvature on training data...")
    measure_curvature(model, train_loader, 
                        data_fraction=args.data_fraction, 
                        batch_size=args.batch_size, 
                        num_power_iter=args.num_power_iter,
                        device=device)

    logger.info("\nEstimating curvature on test data...")
    measure_curvature(model, test_loader, 
                        data_fraction=args.data_fraction, 
                        batch_size=args.batch_size, 
                        num_power_iter=args.num_power_iter,
                        device=device)