"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.u:
                self.u[p] = 0
            grad = self.momentum * self.u[p] + (1 - self.momentum) * (
                p.grad.detach() + self.weight_decay * p.detach()
            )
            self.u[p] = ndl.Tensor(grad, dtype=p.dtype, requires_grad=False)
            p.data -= self.lr * self.u[p]

        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.m:
                self.m[p] = 0
            if p not in self.v:
                self.v[p] = 0
            grad = p.grad.detach() + self.weight_decay * p.detach()
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * grad**2
            m_hat = self.m[p] / (1 - self.beta1**self.t)
            v_hat = self.v[p] / (1 - self.beta2**self.t)
            theta = self.lr * m_hat / (ndl.ops.power_scalar(v_hat, 0.5) + self.eps)
            p.data -= ndl.Tensor(
                theta, dtype=p.dtype, requires_grad=False
            )  # outofmemory now

        ### END YOUR SOLUTION
