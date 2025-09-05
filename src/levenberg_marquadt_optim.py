import torch
from torch.optim.optimizer import Optimizer

class DiagLM(Optimizer):
    """
    Stochastic Diagonal Levenberg–Marquardt
    Update: p <- p - lr * grad / (mu + v)
    where v is an EMA of grad**2 (diag-Hessian proxy as reccomended by LeCun et al. 1998).

    Args:
        params: iterable of parameters
        lr: global step scale epsilon
        mu: damping term mu (keeps steps finite)
        beta: EMA factor for the squared gradients (0.95–0.995 work well)
        weight_decay: L2 penalty added to the grad (coupled)
    """
    def __init__(self, params, lr=1.0, mu=1e-3, beta=0.99, weight_decay=0.0):
        defaults = dict(lr=lr, mu=mu, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group["lr"]
            mu   = group["mu"]
            beta = group["beta"]
            wd   = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Optional L2
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                v = state["v"]
                # v_t = beta * v_{t-1} + (1-beta) * g^2
                v.mul_(beta).addcmul_(grad, grad, value=(1.0 - beta))

                # p <- p - lr * g / (mu + v)
                denom = mu + v
                p.addcdiv_(grad, denom, value=-lr)

        return loss