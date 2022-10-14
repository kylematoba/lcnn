import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from models.layers import ApplyOverHalves, ConvBNBlock
from models.psoftplus import ParametricSoftplus


class CELoss(torch.nn.Module):
    def __init__(self, 
                 model: torch.nn.Module,
                 loss_criterion: Callable = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.metadata = {}
        self.model = model
        self.loss_criterion = loss_criterion

    def compute_loss(self, input, target):
        out = self.model(input)
        raw_loss = self.loss_criterion(out, target)
        self.metadata = {
            'raw_loss': raw_loss.item()
        }
        return raw_loss


def flatten_model(model: torch.nn.Module) -> torch.nn.Sequential:
    flattened_layers = []
    for layer in model:
        if hasattr(layer, "layers"):
            to_add = list(flatten_model(layer.layers))
        elif type(layer) == torch.nn.Sequential:
            to_add = list(flatten_model(layer))
        else:
            to_add = [layer]
        flattened_layers += to_add
    return torch.nn.Sequential(*flattened_layers)


class GradNormRegularizedLoss(CELoss):
    # norm of gradient of out w.r.t. input
    # usually out are logits or log-probabilites
    def __init__(self,
                model: nn.Module, 
                reg_constant: float = 1e-3):
        super().__init__(model)
        self.reg_constant = reg_constant

    def compute_loss(self, input, target):
        out = self.model(input)
        raw_loss = self.loss_criterion(out, target)

        grad_x = torch.autograd.grad(raw_loss, input,
                                     only_inputs=True,
                                     create_graph=True)[0]
        gradnorm = 1e5 * grad_x.pow(2).sum() / input.size(0)

        self.metadata = {
            'raw_loss': raw_loss.item(),
            'gradnorm': gradnorm.item()
        }
        return raw_loss + self.reg_constant * gradnorm


class CURERegularizedLoss(CELoss):
    # norm of gradient of out w.r.t. input
    # usually out are logits or log-probabilites
    def __init__(self,
                model: nn.Module, 
                reg_constant: float = 1e-3):
        super().__init__(model)
        self.reg_constant = reg_constant

    def compute_loss(self, input, target):
        out = self.model(input)
        raw_loss = self.loss_criterion(out, target)

        grad_x = torch.autograd.grad(raw_loss, input, only_inputs=True, create_graph= True)[0]
        gradnorm = 1e5 * grad_x.pow(2).sum() / input.size(0)

        noise = torch.sign(grad_x) / torch.norm(torch.sign(grad_x), p=2, dim=(1, 2, 3), keepdim=True)
        out_noisy = self.model(input + noise.detach())
        raw_loss_noisy = self.loss_criterion(out_noisy, target)
        
        grad_x_noisy = torch.autograd.grad(raw_loss_noisy, input, only_inputs=True, create_graph=True)[0]
        cure_loss = 1e3 * torch.norm(grad_x - grad_x_noisy, p=2).pow(2) / input.size(0)

        self.metadata = {
            'raw_loss': raw_loss.item(),
            'gradnorm': gradnorm.item(),
            'cure_loss': cure_loss.item()
        }
        return raw_loss + self.reg_constant * cure_loss


class CurvatureUpperBoundRegularizedLoss(CELoss):
    # curvature regularization by computing the upper bound
    def __init__(self,
                model: nn.Module,
                reg_constant: float = 1e-2):
        super().__init__(model)
        self.reg_constant = reg_constant

    @torch.no_grad()
    def _apply_hooks_and_init(self, device):
        # add forward hooks        
        self.fwd_handles = []
        for m in self.model.modules():
            if isinstance(m, ApplyOverHalves):
                handle_f = m.register_forward_hook(self._extract_lipschitz_residual)
                self.fwd_handles.append(handle_f)
            elif isinstance(m, ConvBNBlock):
                handle_f = m.register_forward_hook(self._extract_lipschitz)
                self.fwd_handles.append(handle_f) 
            elif isinstance(m, ParametricSoftplus):
                handle_f = m.register_forward_hook(self._extract_curvature)
                self.fwd_handles.append(handle_f)

        self.modules_visited = []
        self.regularizer = [torch.Tensor([1.]).to(device)]
        self.index = 0

    @torch.no_grad()
    def _remove_hooks(self):
        for f in self.fwd_handles:
            f.remove()

    def _compute_regularizer_value(self):
        # compute curvature value by adding layerwise values
        # 1) omit last value
        # 2) subtract max value for numerical stability

        maxval = -math.inf
        for i in range(self.index):
            if maxval < self.regularizer[i]:
                maxval = self.regularizer[i]         

        curvature = 0.
        for i in range(self.index):
            curvature += (self.regularizer[i] - maxval).exp()
        
        self.log_curvature = curvature.log() + maxval

    def _extract_lipschitz_residual(self, module, input, output):
        temp1, temp2 = 1, 1

        if type(module.f1) == nn.Conv2d:
            temp1 = module.f1.log_lipschitz
            self.modules_visited.append(module.f1)

        if type(module.f2) == nn.Conv2d:
            temp2 = module.f2.log_lipschitz
            self.modules_visited.append(module.f2)

        self.regularizer[self.index] += torch.maximum(temp1, temp2)
        
    def _extract_lipschitz(self, module, input, output):
        if module not in self.modules_visited:
            self.regularizer[self.index] += module.log_lipschitz

    def _extract_curvature(self, module, input, output):
        self.regularizer[self.index] += module.log_beta
        self.index += 1
        self.regularizer.append(self.regularizer[self.index-1].clone())

    def compute_loss(self, input, target):
        self._apply_hooks_and_init(input.device)
        out = self.model(input)
        self._compute_regularizer_value()
        self._remove_hooks()

        raw_loss = self.loss_criterion(out, target)

        self.metadata = {
            'raw_loss': raw_loss,
            'log_curvature': self.log_curvature.item()
        }
        return raw_loss + self.reg_constant * self.log_curvature


class CurvatureJVPRegularizedLoss(CELoss):
    # curvature regularization by computing the layerwise "forward AD-style"
    # jacobian vector products using finite differences

    def __init__(self, 
                model: nn.Module,
                noise_std: float = 1e-3,
                reg_constant: float = 1e-2):
        super().__init__(model)

        self.noise_std = noise_std
        self.reg_constant = reg_constant

    @torch.no_grad()
    def _apply_hooks_and_init(self):
        # add forward hooks        
        self.fwd_handles = []
        for m in self.model.modules():
            if isinstance(m, ParametricSoftplus):
                handle_f = m.register_forward_hook(self._extract_curvature)
                self.fwd_handles.append(handle_f)

        self.modules_visited = []
        self.betas = []
        self.features = []
        self.index = 0

    @torch.no_grad()
    def _remove_hooks(self):
        for f in self.fwd_handles:
            f.remove()

    def _compute_regularizer_value(self, 
                        betas: list, 
                        orig_features: list, 
                        noisy_features: list):
        # compute curvature value by adding layerwise values
        self.layerwise_curvatures = []
        self.curvature = 0.
        for i in range(self.index):
            temp = betas[i] * (noisy_features[i] - orig_features[i]).pow(2).sum().sqrt()
            self.layerwise_curvatures.append(temp.data.item())
            self.curvature += temp

    def _extract_curvature(self, module, input, output):
        self.features.append(input[0])
        self.betas.append(module.log_beta.exp())
        self.index += 1

    def compute_loss(self, input, target):
        self._apply_hooks_and_init()
        out = self.model(input)     
        self._remove_hooks()

        raw_loss = self.loss_criterion(out, target)
        loss_grad = torch.autograd.grad(outputs=raw_loss, inputs=input, retain_graph=True)[0]
        loss_grad /= loss_grad.norm(p=2, dim=(1,2,3), keepdim=True)

        # save features in a separate list
        orig_features = self.features.copy()

        self._apply_hooks_and_init()
        #noise = self.noise_std * torch.randn_like(input)
        noise = (self.noise_std * loss_grad.detach_())#.requires_grad_()
        self.model(input + noise)
        self._remove_hooks()

        self._compute_regularizer_value(betas = self.betas, 
                                        orig_features = orig_features, 
                                        noisy_features = self.features)

        self.metadata = {
            'raw_loss': raw_loss.item(),
            'curvature': self.curvature.data.item()
        }
        return raw_loss + self.reg_constant * self.curvature
        

class CurvatureProxyRegularizedLoss(CELoss):
    # curvature regularization by computing the proxy
    def __init__(self,
                model: nn.Module, 
                reg_constants: Tuple[float, float] = (1e-2, 1e-2)):
        super().__init__(model)
        self.reg_constants = reg_constants

    @torch.no_grad()
    def _apply_hooks_and_init(self, device):
        # add forward hooks        
        self.fwd_handles = []
        for m in self.model.modules():
            if isinstance(m, ApplyOverHalves):
                handle_f = m.register_forward_hook(self._extract_lipschitz_residual)
                self.fwd_handles.append(handle_f)
            elif isinstance(m, ConvBNBlock):
                handle_f = m.register_forward_hook(self._extract_lipschitz)
                self.fwd_handles.append(handle_f) 
            elif isinstance(m, ParametricSoftplus):
                handle_f = m.register_forward_hook(self._extract_curvature)
                self.fwd_handles.append(handle_f)

        self.modules_visited = []
        self.betas = []
        self.gammas = []
        self.index = 0

    @torch.no_grad()
    def _remove_hooks(self):
        for f in self.fwd_handles:
            f.remove()

    def _extract_lipschitz_residual(self, module, input, output):
        temp1, temp2 = 1, 1

        if type(module.f1) == nn.Conv2d:
            temp1 = module.f1.log_lipschitz
            self.modules_visited.append(module.f1)

        if type(module.f2) == nn.Conv2d:
            temp2 = module.f2.log_lipschitz
            self.modules_visited.append(module.f2)

        self.gammas.append(torch.maximum(temp1, temp2))
        
    def _extract_lipschitz(self, module, input, output):
        if module not in self.modules_visited:
            self.gammas.append(module.log_lipschitz)

    def _extract_curvature(self, module, input, output):
        self.betas.append(module.log_beta.exp())

    def compute_loss(self, input, target):
        self._apply_hooks_and_init(input.device)
        out = self.model(input)
        self._remove_hooks()

        beta_term = 0.
        for b in self.betas:
            beta_term += b
        
        if len(self.gammas) > 0:
            gamma_term = 0.
            for g in self.gammas:
                gamma_term += g.abs()
            
        raw_loss = self.loss_criterion(out, target)

        self.metadata = {
            'raw_loss': raw_loss,
            'beta_term': beta_term.item(),
            'gamma_term': gamma_term.item()
        }

        if len(self.gammas) > 0:
            self.metadata.update({'gamma term:': gamma_term.item()})
            return raw_loss + self.reg_constants[0] * beta_term + self.reg_constants[1] * gamma_term
        else:
            return raw_loss + self.reg_constants[0] * beta_term 


class CurvatureProxyAndGradientRegularizedLoss(CELoss):
    # curvature regularization by computing the proxy
    # also include gradient norm regularization

    def __init__(self,
                model: nn.Module, 
                reg_constants: Tuple[float, float, float] = (1e-2, 1e-2, 1e-2)):
        super().__init__(model)
        self.reg_constants = reg_constants

    @torch.no_grad()
    def _apply_hooks_and_init(self, device):
        # add forward hooks        
        self.fwd_handles = []
        for m in self.model.modules():
            if isinstance(m, ApplyOverHalves):
                handle_f = m.register_forward_hook(self._extract_lipschitz_residual)
                self.fwd_handles.append(handle_f)
            elif isinstance(m, ConvBNBlock):
                handle_f = m.register_forward_hook(self._extract_lipschitz)
                self.fwd_handles.append(handle_f) 
            elif isinstance(m, ParametricSoftplus):
                handle_f = m.register_forward_hook(self._extract_curvature)
                self.fwd_handles.append(handle_f)

        self.modules_visited = []
        self.betas = []
        self.gammas = []
        self.index = 0

    @torch.no_grad()
    def _remove_hooks(self):
        for f in self.fwd_handles:
            f.remove()

    def _extract_lipschitz_residual(self, module, input, output):
        temp1, temp2 = 1, 1

        if type(module.f1) == nn.Conv2d:
            temp1 = module.f1.log_lipschitz
            self.modules_visited.append(module.f1)

        if type(module.f2) == nn.Conv2d:
            temp2 = module.f2.log_lipschitz
            self.modules_visited.append(module.f2)

        self.gammas.append(torch.maximum(temp1, temp2))
        
    def _extract_lipschitz(self, module, input, output):
        if module not in self.modules_visited:
            self.gammas.append(module.log_lipschitz)

    def _extract_curvature(self, module, input, output):
        self.betas.append(module.log_beta.exp())

    def compute_loss(self, input, target):
        self._apply_hooks_and_init(input.device)
        out = self.model(input)
        self._remove_hooks()

        beta_term = 0.
        for b in self.betas:
            beta_term += b
        
        if len(self.gammas) > 0:
            # gamma_term =  torch.tensor(0.)
            gamma_term = 0.
            for g in self.gammas:
                gamma_term += g.abs()
            
        raw_loss = self.loss_criterion(out, target)
        loss_grad = torch.autograd.grad(outputs=raw_loss,
                                        inputs=input,
                                        create_graph=True)[0]
        gradnorm = 1e5 * loss_grad.pow(2).sum() / input.size(0)
        
        self.metadata = {
            'raw_loss': raw_loss,
            'beta_term': beta_term.item() if type(beta_term) == torch.Tensor else beta_term,
            'gamma_term': gamma_term.item() if type(gamma_term) == torch.Tensor else gamma_term,
            'gradnorm': gradnorm.item()
        }

        if len(self.gammas) > 0:
            self.metadata.update({'gamma term:': gamma_term.item()})
            return raw_loss + self.reg_constants[0] * beta_term  \
                            + self.reg_constants[1] * gamma_term \
                            + self.reg_constants[2] * gradnorm
        else:
            return raw_loss + self.reg_constants[0] * beta_term \
                            + self.reg_constants[2] * gradnorm
