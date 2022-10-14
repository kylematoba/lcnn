import warnings

import torch


class ParametricSoftplus(torch.nn.Module):
    # Adapted from:
    # https://discuss.pytorch.org/t/learnable-parameter-in-softplus/60371/2
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#PReLU
    def __init__(self,
                 init_beta: float = 5.,
                 threshold: float = 20.0):
        super().__init__()
        assert init_beta > 0.0
        assert threshold >= 0.0
        if 0.0 == threshold:
            warnings.warn("This is simply going to be relu")

        # parameterize in terms of log in order to keep beta > 0
        self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log())
        self.threshold = threshold
        self.register_buffer('offset', torch.log(torch.tensor(2.)), persistent=False)
        self.eps = 1e-3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
        beta = self.log_beta.exp()
        beta_x = (beta + self.eps) * x
        y = (torch.nn.functional.softplus(beta_x, beta=1.0, threshold=self.threshold) - self.offset) / (beta + self.eps)
        return y
