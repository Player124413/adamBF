import torch
from torch.optim import AdamW
import logging
from math import sqrt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdamBF(AdamW):
    """
    AdamBF is a variant of the AdamW optimizer with AdaBelief enabled by default.
    AdaBelief adapts the step size by considering the belief in the observed gradients,
    which can lead to better performance than traditional Adam variants.
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator for numerical stability (default: 1e-16).
        weight_decay (float, optional): Weight decay coefficient (default: 0).
        weight_decouple (bool, optional): Whether to decouple weight decay (AdamW style) (default: True).
        rectify (bool, optional): Whether to use rectification as in RAdam (default: False).
        debug (bool, optional): Whether to enable debug logging (default: True).
    """
    debug = True
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, 
                 weight_decay=0, weight_decouple=True, rectify=False, debug=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        weight_decouple=weight_decouple, rectify=rectify)
        super(AdamW, self).__init__(params, defaults)
        
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using AdaBelief.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            
        Returns:
            loss: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamBF does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of (grad - exp_avg)^2
                    state['exp_avg_var'] = torch.zeros_like(p)
                    if group['rectify']:
                        # Length of approximated SMA
                        state['rho_inf'] = 2 / (1 - group['betas'][1]) - 1
                
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                
                if group['weight_decouple']:
                    # AdamW style weight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    denom = (exp_avg_var.sqrt() / sqrt(bias_correction2)).add_(group['eps'])
                else:
                    # Adam style weight decay
                    denom = (exp_avg_var.sqrt() / sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                if group['rectify']:
                    # Rectification as in RAdam
                    rho_inf = state['rho_inf']
                    rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])
                    if rho_t > 4:
                        # Variance rectification term
                        r_t = sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        step_size = step_size * r_t
                    else:
                        step_size = step_size * 0
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                if not group['weight_decouple'] and group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                if self.debug:
                    logger.debug(f"Parameter: {p.shape}")
                    logger.debug(f"Learning rate: {group['lr']}")
                    logger.debug(f"Step size: {step_size}")
                    logger.debug(f"Gradient norm: {torch.norm(grad).item():.6f}")
                    logger.debug(f"Exp_avg norm: {torch.norm(exp_avg).item():.6f}")
                    logger.debug(f"Exp_avg_var norm: {torch.norm(exp_avg_var).item():.6f}")
                    logger.debug(f"Denominator norm: {torch.norm(denom).item():.6f}")
                    if group['rectify'] and 'rho_t' in locals():
                        logger.debug(f"Rectification term (r_t): {r_t if rho_t > 4 else 0:.6f}")

        return loss
