import torch
from torch.optim import AdamW

class AdamBF(AdamW):
    """
    AdamBF is a variant of the AdamW optimizer with amsgrad enabled by default.
    This makes it more stable in certain scenarios by ensuring that the learning rate
    does not increase over time, addressing potential convergence issues.
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator for numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay coefficient (default: 0).
        amsgrad (bool, optional): Whether to use the AMSGrad variant (default: True).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True):
        super(AdamBF, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
