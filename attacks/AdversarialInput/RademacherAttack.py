import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker


__all__ = ["MI_RademacherAttack", "Rademacher_CommonWeakness"]


class MI_RademacherAttack(AdversarialInputAttacker):
    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = False,
        step_size: float = 1 / 255,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        mu: float = 1,
        rademacher_range: float = 8 / 255,
        rademacher_iterations: int = 10,
        *args,
        **kwargs
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.rademacher_range = rademacher_range
        self.rademacher_iterations = rademacher_iterations
        super().__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            for model in self.models:
                for _ in range(self.rademacher_iterations):
                    now_x = x + torch.randn_like(momentum).sign() * self.rademacher_range
                    logit = model(now_x.to(model.device)).to(x.device)
                    loss = self.criterion(logit, y)
                    loss.backward()
            grad = x.grad / self.rademacher_iterations
            x.requires_grad = False
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = self.clamp(x, original_x)
        return x


class Rademacher_CommonWeakness(AdversarialInputAttacker):
    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = False,
        step_size: float = 16 / 255 / 5,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        mu=1,
        inner_step_size: float = 50,
        rademacher_range: float = 32 / 255,
        rademacher_iterations: int = 10,
        *args,
        **kwargs
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(Rademacher_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size
        self.rademacher_range = rademacher_range
        self.rademacher_iterations = rademacher_iterations

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.grad = None
            self.begin_attack(x.clone().detach())
            for model in self.models * self.rademacher_iterations:
                x.requires_grad = True
                aug_x = x + torch.randn_like(inner_momentum).sign() * self.rademacher_range
                loss = self.criterion(model(aug_x.to(model.device)), y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1
                    )
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1
                    )
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        """
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        """
        fake_grad = now - self.original  # x_n-x
        self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
        now.mul_(0)
        now.add_(self.original)
        now.add_(ksi * self.outer_momentum.sign())
        now = clamp(now)
        del self.grad_record
        del self.original
        return now
