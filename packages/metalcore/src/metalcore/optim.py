
import torch
from torch.optim import Optimizer

try:
    import metalcore_backend
except ImportError:
    metalcore_backend = None

import math

class MetalAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if amsgrad:
            raise ValueError("MetalAdamW does not support amsgrad yet")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MetalAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is not None:
                    # Generic handling
                    if p.grad.is_sparse:
                        raise RuntimeError('MetalAdamW does not support sparse gradients')
                        
                    # State initialization
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        # CRITICAL: Always keep optimizer states in float32 for numerical stability
                        # This is essential for bf16/fp16 training - states accumulate small values
                        # that would underflow in reduced precision
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                    
                    state['step'] += 1
                    
                    # Metal dispatch
                    if p.is_mps and metalcore_backend is not None and p.is_contiguous() and p.grad.is_contiguous():
                        # Calculate corrections
                        bias_correction1 = 1 - beta1 ** state['step']
                        bias_correction2 = 1 - beta2 ** state['step']
                        
                        metalcore_backend.adamw_step(
                            p, 
                            p.grad, 
                            state['exp_avg'], 
                            state['exp_avg_sq'],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['weight_decay'],
                            bias_correction1,
                            bias_correction2
                        )
                    else:
                        # Fallback (manual python loop or single tensor)
                        # This matches PyTorch's functional_adamw roughly
                        grad = p.grad
                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']
                        step = state['step']
                        
                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step
                        
                        # Weight decay
                        if group['weight_decay'] != 0:
                            p.mul_(1 - group['lr'] * group['weight_decay'])
                            
                        # Decay the first and second moment running average coefficient
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                        step_size = group['lr'] / bias_correction1
                        
                        p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
