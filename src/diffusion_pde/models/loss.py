import torch
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def __call__(self, net, x, t):
        pass


class EDMLoss(Loss):
    '''
    taken from "elucidating the design space..." paper:
    https://github.com/NVlabs/edm/blob/main/training/loss.py
    '''
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, reduce=True, reduce_method="mean"):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.reduce = reduce
        self.reduce_method = reduce_method

    def __call__(self, net, x, t):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        D_yn = net(x + n, sigma.flatten(), t)
        loss = weight * ((D_yn - x) ** 2)

        if not self.reduce:
            return loss

        if self.reduce_method == "mean":
            return loss.mean(dim=(1,2,3))
        elif self.reduce_method == "sum":
            return loss.sum(dim=(1,2,3))