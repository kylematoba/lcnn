import logging
from typing import Tuple

import torch

import utils.logging


# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(utils.logging.get_standard_streamhandler())


def smoothness_test_exact(model: torch.nn.Module,
                          image: torch.Tensor,
                          target: torch.Tensor):
    # A smoothness test using exact computation of curvature
    model.eval()
    # num_power_iter = 20
    num_power_iter = 4
    u = torch.randn_like(image)
    u /= torch.norm(u, p=2, dim=(1, 2, 3), keepdim=True)

    with torch.enable_grad():
        image = image.requires_grad_()
        out = model(image)
        y = F.log_softmax(out, 1)
        orig_output = F.nll_loss(y, target, reduction='none')
        model.zero_grad()
        # Gradients w.r.t. input
        gradients = torch.autograd.grad(outputs=orig_output.sum(),
                                        inputs=image, create_graph=True)[0]
        gnorm = torch.norm(gradients, p=2, dim=(1, 2, 3))
        assert not gradients.isnan().any()

        for _ in range(num_power_iter):
            # print(_)
            grad_vector_prod = (gradients * u.detach_()).sum()
            hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image, retain_graph=True)[0]
            assert not hessian_vector_prod.isnan().any()

            hvp_norm = torch.norm(hessian_vector_prod, p=2, dim=(1, 2, 3), keepdim=True)
            u = torch.where(hvp_norm > 0, hessian_vector_prod / hvp_norm, 0.0)
            assert not u.isnan().any(), "nan element in u"
        grad_vector_prod = (gradients * u.detach_()).sum()
        hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image)[0]
        hessian_singular_value = (hessian_vector_prod * u.detach_()).sum((1, 2, 3))
    curvatures = torch.where(gnorm == 0, 0.0, hessian_singular_value.abs().div(gnorm))
    hess = hessian_singular_value.abs()
    grad = gnorm
    return curvatures, hess, grad


def evaluate_curvature(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       max_batches: int,
                       batch_size: int,
                       device: torch.device) -> Tuple[tuple, tuple, tuple]:
    model.eval()
    datasize = max_batches * batch_size
    curvature_agg = torch.empty(size=(datasize,))
    grad_agg = torch.empty(size=(datasize,))
    hess_agg = torch.empty(size=(datasize,))

    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device).requires_grad_(), target.to(device)
        with torch.no_grad():
            curvatures, hess, grad = smoothness_test_exact(model, data, target)
        curvature_agg[idx * batch_size:(idx + 1) * batch_size] = curvatures.detach()
        hess_agg[idx * batch_size:(idx + 1) * batch_size] = hess.detach()
        grad_agg[idx * batch_size:(idx + 1) * batch_size] = grad.detach()

        avg_curvature, std_curvature = curvature_agg.mean().item(), curvature_agg.std().item()
        avg_hessian, std_hessian = hess_agg.mean().item(), hess_agg.std().item()
        avg_grad, std_grad = grad_agg.mean().item(), grad_agg.std().item()

        if idx == (max_batches - 1):
            # logger.info('Average Curvature: {:.4f} +/- {:.4f} '.format(avg_curvature, std_curvature))
            # logger.info('Average Hessian Spectral Norm: {:.4f} +/- {:.4f} '.format(avg_hessian, std_hessian))
            # logger.info('Average Gradient Norm: {:.4f} +/- {:.4f}'.format(avg_grad, std_grad))
            return (avg_curvature, std_curvature), (avg_hessian, std_hessian), (avg_grad, std_grad)


if __name__ == "__main__":
    # https://t.co/dZ0bITXAXr
    pass
