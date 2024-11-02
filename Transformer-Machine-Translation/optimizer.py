import torch
import torch.nn as nn


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1

    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class Adam(nn.Module):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.params = list(params)

        self.param_step = {p: 0 for p in self.params}
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}

        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            self.update_param(p)

    def update_param(self, p):
        self.param_step[p] += 1

        self.param_momentum[p] = (
            1 - self.beta1
        ) * p.grad + self.beta1 * self.param_momentum[p]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (
            p.grad
        ) ** 2 + self.beta2 * self.param_2nd_momentum[p]

        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        p_2nd_momentum = self.param_2nd_momentum[p] / bias_correction_2
        p_momentum = self.param_momentum[p] / bias_correction_1

        p_lr = self.lr / (torch.sqrt(p_2nd_momentum) + self.eps)
        p_update = -p_lr * p_momentum

        p.data.add_(p_update)
