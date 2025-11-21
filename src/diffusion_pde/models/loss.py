import torch
from abc import ABC, abstractmethod
from .pde_losses import X_and_dXdt_fd


class Loss(ABC):
    @abstractmethod
    def __call__(self, net, x, t):
        pass


class EDMLoss(Loss):
    '''
    taken from "elucidating the design space..." paper:
    https://github.com/NVlabs/edm/blob/main/training/loss.py
    '''
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, reduce_method="mean"):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.reduce_method = reduce_method

    def __call__(self, net, x, *args, **kwargs):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x, device=x.device) * sigma
        D_yn = net(x + n, sigma.flatten(), *args, **kwargs)
        loss = weight * ((D_yn - x) ** 2)

        if self.reduce_method == "mean":
            return loss.mean(dim=(1,2,3))
        elif self.reduce_method == "sum":
            return loss.sum(dim=(1,2,3))


class EDMPhysicsLoss(Loss):
    def __init__(self, 
        pde_loss_fn, 
        pde_loss_kwargs,
        pde_loss_coeff=1.0, 
        ch_a=0,
        P_mean=-1.2, 
        P_std=1.2, 
        sigma_data=0.5, 
        reduce_method="mean"
        ):
        self.pde_loss_fn = pde_loss_fn
        self.pde_loss_kwargs = pde_loss_kwargs
        self.pde_loss_coeff = pde_loss_coeff
        self.ch_a = ch_a
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.reduce_method = reduce_method

    def __call__(self, net, x, labels, **kwargs):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x, device=x.device) * sigma
        D_yn, dxdt = X_and_dXdt_fd(net, x + n, sigma.flatten(), labels, **kwargs)
        dxdt = dxdt.detach()
        dxdt = dxdt[:, self.ch_a:, ...]  # select channel for PDE loss
        #D_yn = net(x + n, sigma.flatten(), *args, **kwargs)
        edm_loss = weight * ((D_yn - x) ** 2)

        if self.reduce_method == "mean":
            edm_loss = edm_loss.mean(dim=(1,2,3))
        elif self.reduce_method == "sum":
            edm_loss = edm_loss.sum(dim=(1,2,3))

        pde_loss = self.pde_loss_fn(D_yn[:, self.ch_a:, ...], dxdt, labels, **self.pde_loss_kwargs)

        loss = edm_loss + self.pde_loss_coeff * pde_loss
        
        return loss