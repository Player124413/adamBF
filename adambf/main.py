import torch
from torch.optim import AdamW
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        debug (bool, optional): Whether to enable debug logging (default: True).
    """
    debug=True
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True):
        super(AdamBF, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value if closure is provided, else None.
        """
        loss = super(AdamBF, self).step(closure)

        if self.debug:
            for group in self.param_groups:
                lr = group['lr']
                weight_decay = group['weight_decay']
                logger.debug(f"Learning rate: {lr}, Weight decay: {weight_decay}")
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_norm = torch.norm(p.grad).item()
                    param_norm = torch.norm(p).item()
                    logger.debug(f"Parameter norm: {param_norm:.6f}, Gradient norm: {grad_norm:.6f}")
                    
                    # Log first moment (mean) and second moment (uncentered variance) if available
                    state = self.state[p]
                    if 'exp_avg' in state:
                        exp_avg_norm = torch.norm(state['exp_avg']).item()
                        logger.debug(f"First moment (exp_avg) norm: {exp_avg_norm:.6f}")
                    if 'exp_avg_sq' in state:
                        exp_avg_sq_norm = torch.norm(state['exp_avg_sq']).item()
                        logger.debug(f"Second moment (exp_avg_sq) norm: {exp_avg_sq_norm:.6f}")
                    if self.defaults['amsgrad'] and 'max_exp_avg_sq' in state:
                        max_exp_avg_sq_norm = torch.norm(state['max_exp_avg_sq']).item()
                        logger.debug(f"Max second moment (max_exp_avg_sq) norm: {max_exp_avg_sq_norm:.6f}")

        return loss
