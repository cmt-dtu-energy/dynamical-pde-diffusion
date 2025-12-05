import torch
import logging
import numpy as np
from torch.func import jvp, vmap
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class no_op:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass

def X_and_dXdt_fd(net, x, sigma, labels, eps=1e-5, no_grad=True, **kwargs):
    """
    Compute the output of the network and its derivative with respect to time t.
    First parameter of labels must be time t.
    Finite difference approximation to check correctness of jvp implementation.
    Parameters
    ----------
    net : callable
        The neural network model that takes inputs (x, sigma, t).
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    sigma : torch.Tensor
        Noise level tensor of shape (B,).
    labels : torch.Tensor
        labels tensor of shape (B, label_dim).
    eps : float, optional
        Finite difference step size, by default 1e-5.   

    Returns
    -------
    X : torch.Tensor
        Output of the network of shape (B, C, H, W).
    dXdt : torch.Tensor
        Derivative of the output with respect to time t, of shape (B, C, H, W).
    """

    if labels is None:
        u0 = net(x, sigma, labels, **kwargs)
        dudt_fd = torch.zeros_like(u0)
        return u0, dudt_fd
    
    lbl_p = labels.detach().clone(); lbl_m = labels.detach().clone()
    lbl_p[:, 0] += eps
    lbl_m[:, 0] -= eps

    if no_grad:
        ctx = torch.no_grad()
    else:
        ctx = no_op()
    with ctx:
        up = net(x, sigma, lbl_p, **kwargs)
        um = net(x, sigma, lbl_m, **kwargs)
        
    dudt_fd = (up - um) / (2 * eps)
    
    u0 = net(x, sigma, labels, **kwargs)
    del up, um, lbl_p, lbl_m
    return u0, dudt_fd


def X_and_dXdt(net, x, sigma, labels):
    """
    Compute the output of the network and its derivative with respect to time t.
    First parameter of labels must be time t.
    Parameters
    ----------
    net : callable
        The neural network model that takes inputs (x, sigma, t).
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    sigma : torch.Tensor
        Noise level tensor of shape (B,).
    labels : torch.Tensor
        labels tensor of shape (B, label_dim).

    Returns 
    -------
    X : torch.Tensor
        Output of the network of shape (B, C, H, W).
    dXdt : torch.Tensor
        Derivative of the output with respect to time t, of shape (B, C, H, W).
    """
    t0 = labels[:, 0]

    def f(t):
        lbl = labels.clone()
        lbl[:, 0] = t
        return net(x, sigma, lbl)
    
    X, dXdt = jvp(
        f,
        (t0,),
        (torch.ones_like(t0),)
    )
    return X, dXdt


def laplacian(u, dx):
    """
    Computes the Laplacian of a 2D tensor according to 
        ∇²u[i, j] ≈ (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4u[i, j]) / (dx^2)
    where dx is the grid spacing.
    For more details see
    https://en.wikipedia.org/wiki/Finite_difference_method#Example:_The_Laplace_operator

    Parameters
    ----------
    u : torch.Tensor
        Input tensor of shape (B, C, H, W).
    dx : float
        Grid spacing.

    Returns
    -------
    torch.Tensor
        Laplacian of u, of shape (B, C, H, W).
    """
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=u.dtype, device=u.device
    ).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 3, 3)
    
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode='reflect')  # shape (B, C, H+2, W+2)
    laplacian_u = torch.nn.functional.conv2d(u_padded, laplacian_kernel) / (dx ** 2)  # shape (B, C, H, W)
    return laplacian_u


class Sampler(ABC):
    """
    Sampler class for EDM style sampling.
    
    Parameters
    ----------
    net : callable
        The neural network model that takes inputs (x, sigma, t).
    device : torch.device
        Device to run the sampler on.
    sample_shape : tuple
        Shape of the samples to generate (H, W).
    num_steps : int, optional
        Number of sampling steps, by default 18.
    sigma_min : float, optional
        Minimum noise level, by default 0.002.
    sigma_max : float, optional
        Maximum noise level, by default 80.0.
    rho : float, optional
        Rho parameter for noise schedule, by default 7.0.
    """
    def __init__(
        self,
        net,
        device,
        sample_shape,
        num_channels,
        num_samples,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        out_and_grad_fn=X_and_dXdt_fd,
    ):
        self.net = net
        self.device = device
        self.sample_shape = sample_shape
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.out_and_grad_fun = out_and_grad_fn

        self.dtype_f = torch.float32     # net runs in fp32
        self.dtype_t = torch.float64     # keep time grid in fp64 for stability, as in EDM

    @torch.no_grad()
    def sample(
        self,
        labels=None,
        net_obs=None,
        num_steps=None,     # number of sampling steps
        sigma_min=None,     # minimum sigma
        sigma_max=None,     # maximum sigma
        rho=None,           # rho parameter for noise schedule
    ):  
        dt_f = self.dtype_f
        dt_t = self.dtype_t

        num_steps = num_steps if num_steps is not None else self.num_steps
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max
        rho = rho if rho is not None else self.rho

        step_idx = torch.arange(num_steps, dtype=dt_t, device=self.device)
        sigmas = (sigma_max**(1.0/rho) + step_idx/(num_steps-1) * (sigma_min**(1.0/rho) - sigma_max**(1.0/rho)))**rho
        sigmas = getattr(self.net, "round_sigma", lambda x: x)(sigmas)
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # length N+1, last = 0

        B = labels.shape[0] if labels is not None else self.num_samples
        if labels is not None:
            labels = labels.to(device=self.device, dtype=dt_f)
        if net_obs is not None:
            net_obs = net_obs.to(device=self.device, dtype=dt_f)
            args = (labels, net_obs)
        else:
            args = (labels,)

        latents = torch.randn((B, self.num_channels, *self.sample_shape), device=self.device, dtype=dt_t)

        x_next = (latents.to(self.dtype_t) * sigmas[0])

        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):  # i = 0..N-1
            x_cur = x_next
            # Euler step to t_next
            x_N, dxdt = self.out_and_grad_fun(self.net, x_cur.to(dt_f), torch.full((B,), sigma_cur, device=self.device, dtype=dt_f), *args)
            x_N, dxdt = x_N.to(dt_t), dxdt.to(dt_t)
            
            d_cur = (x_cur - x_N) / sigma_cur
            x_next = x_cur + (sigma_next - sigma_cur) * d_cur
            # Heun (2nd-order) correction unless final step
            if i < num_steps - 1:
                x_N, dxdt = self.out_and_grad_fun(self.net, x_next.to(dt_f), torch.full((B,), sigma_next, device=self.device, dtype=dt_f), *args)
                x_N, dxdt = x_N.to(dt_t), dxdt.to(dt_t)
                d_prime = (x_next - x_N) / sigma_next
                x_next = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(dt_f).detach().cpu()
        

    @abstractmethod
    def sample_joint(self, *args, **kwargs):
        ...

    @abstractmethod
    def sample_forward(self, *args, **kwargs):
        ...

    
class EDMHeatSampler(Sampler):
    def __init__(
        self,
        net,
        device,
        sample_shape,
        num_channels,
        num_samples,
        dx,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ):
        super().__init__(
            net,
            device,
            sample_shape,
            num_channels,
            num_samples,
            num_steps,
            sigma_min,
            sigma_max,
            rho,
    )
        self.dx = dx
    

    def sample_conditional(
        self,
        labels=None,             # (B, label_dim) extra conditioning your Unet expects; use zeros if None
        net_obs=None,       # if network accepts obs as input
        obs_a=None,         # observations of initial condition
        obs_u=None,         # observations of solution at time T
        mask_a=None,        # masks for obs_a
        mask_u=None,        # masks for obs_u
        zeta_a=None,        # weight for obs_a loss
        zeta_u=None,        # weight for obs_u loss
        zeta_pde=None,      # weight for PDE loss
        num_steps=None,     # number of sampling steps
        sigma_min=None,     # minimum sigma
        sigma_max=None,     # maximum sigma
        rho=None,           # rho parameter for noise schedule
        return_losses=False, # whether to return losses for debugging
    ):
        if self.num_channels == 2:   
            return self.sample_joint(
                labels,
                obs_a,
                obs_u,
                mask_a,
                mask_u,
                zeta_a,
                zeta_u,
                zeta_pde,
                num_steps,
                sigma_min,
                sigma_max,
                rho,
                return_losses,
            )
        else:
            return self.sample_forward(
                labels,
                obs_u,
                mask_u,
                zeta_u,
                zeta_pde,
                net_obs,
                num_steps,
                sigma_min,
                sigma_max,
                rho,
                return_losses,
            )
        
    
    def sample_joint(
        self,
        labels,             # (B, label_dim) extra conditioning your Unet expects, None for unconditional
        obs_a,         # observations of initial condition
        obs_u,         # observations of solution at time T
        mask_a,        # masks for obs_a
        mask_u,        # masks for obs_u
        zeta_a,        # weight for obs_a loss
        zeta_u,        # weight for obs_u loss
        zeta_pde,      # weight for PDE loss
        num_steps=None,     # number of sampling steps
        sigma_min=None,     # minimum sigma
        sigma_max=None,     # maximum sigma
        rho=None,           # rho parameter for noise schedule
        return_losses=False, # whether to return losses for debugging
    ):
        dt_f = self.dtype_f
        dt_t = self.dtype_t

        obs_u, mask_u = obs_u.to(device=self.device, dtype=dt_t), mask_u.to(device=self.device, dtype=dt_t)
        obs_a, mask_a = obs_a.to(device=self.device, dtype=dt_t), mask_a.to(device=self.device, dtype=dt_t)

        num_steps = num_steps if num_steps is not None else self.num_steps
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max
        rho = rho if rho is not None else self.rho

        step_idx = torch.arange(num_steps, dtype=dt_t, device=self.device)
        sigmas = (sigma_max**(1.0/rho) + step_idx/(num_steps-1) * (sigma_min**(1.0/rho) - sigma_max**(1.0/rho)))**rho
        sigmas = getattr(self.net, "round_sigma", lambda x: x)(sigmas)
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # length N+1, last = 0

        B = labels.shape[0] if labels is not None else self.num_samples
        if labels is not None:
            labels = labels.to(device=self.device, dtype=dt_f)
        
        latents = torch.randn((B, self.num_channels, *self.sample_shape), device=self.device, dtype=dt_t)

        x_next = (latents.to(self.dtype_t) * sigmas[0])

        losses = torch.zeros((num_steps, 4))  # for debugging

        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):  # i = 0..N-1
            x_cur = x_next.detach().clone()
            x_cur.requires_grad = True
            # Euler step to t_next
            x_N, dxdt = self.out_and_grad_fun(self.net, x_cur.to(dt_f), torch.full((B,), sigma_cur, device=self.device, dtype=dt_f), labels)
            x_N, dxdt = x_N.to(dt_t), dxdt.to(dt_t)
            
            d_cur = (x_cur - x_N) / sigma_cur
            x_next = x_cur + (sigma_next - sigma_cur) * d_cur
            # Heun (2nd-order) correction unless final step
            if i < num_steps - 1:
                x_N, dxdt = self.out_and_grad_fun(self.net, x_next.to(dt_f), torch.full((B,), sigma_next, device=self.device, dtype=dt_f), labels)
                x_N, dxdt = x_N.to(dt_t), dxdt.to(dt_t)
                d_prime = (x_next - x_N) / sigma_next
                x_next = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)

            # observation losses for DPS sampling
            loss_u = torch.zeros(1, dtype=dt_t, device=self.device)
            loss_a = torch.zeros(1, dtype=dt_t, device=self.device)
            if mask_u.sum() > 0:
                loss_u = torch.sqrt(torch.sum((mask_u * (x_N[:, 1:, :, :] - obs_u)) ** 2))
            if mask_a.sum() > 0:
                loss_a = torch.sqrt(torch.sum((mask_a * (x_N[:, :1, :, :] - obs_a)) ** 2))

            # PDE loss for DPS sampling
            lap_u = laplacian(x_N[:, 1:, :, :], self.dx)
            alphas = labels[:, -1].to(dt_t).view(-1, 1, 1, 1) if labels is not None else 1.0
            dudt = dxdt[:, 1:, :, :]
            loss_pde = torch.sqrt(torch.sum((dudt - alphas * lap_u)**2)) / (self.sample_shape[-2] * self.sample_shape[-1])  # normalize by spatial size
            
            if i <= 0.8 * num_steps:
                w_a, w_u, w_pde = zeta_a, zeta_u, zeta_pde
            else:
                w_a, w_u, w_pde = 0.1 * zeta_a, 0.1 * zeta_u, zeta_pde
            
            loss_comb = w_a * loss_a + w_u * loss_u + w_pde * loss_pde
            grad_x = torch.autograd.grad(loss_comb, x_cur, retain_graph=False)[0]
            x_next = x_next - grad_x
            
            losses[i] = torch.tensor([loss_a.item(), loss_u.item(), loss_pde.item(), loss_comb.item()])        

        # Return at sigma=0 in fp32
        x = x_next.to(dt_f).detach().cpu()

        losses = losses.detach().cpu().numpy() if return_losses else None        
        return x, losses
    

    def sample_forward(
        self,
        labels,             # (B, label_dim) extra conditioning your Unet expects; use zeros if None
        obs_u,         # observations of solution at time T
        mask_u,        # masks for obs_u
        zeta_u,        # weight for obs_u loss
        zeta_pde,      # weight for PDE loss
        net_obs=None,       # if network accepts obs as input
        num_steps=None,     # number of sampling steps
        sigma_min=None,     # minimum sigma
        sigma_max=None,     # maximum sigma
        rho=None,           # rho parameter for noise schedule
        return_losses=False, # whether to return losses for debugging
    ):
        dt_f = self.dtype_f
        dt_t = self.dtype_t

        obs_u, mask_u = obs_u.to(device=self.device, dtype=dt_t), mask_u.to(device=self.device, dtype=dt_t)
        
        num_steps = num_steps if num_steps is not None else self.num_steps
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max
        rho = rho if rho is not None else self.rho

        step_idx = torch.arange(num_steps, dtype=dt_t, device=self.device)
        sigmas = (sigma_max**(1.0/rho) + step_idx/(num_steps-1) * (sigma_min**(1.0/rho) - sigma_max**(1.0/rho)))**rho
        sigmas = getattr(self.net, "round_sigma", lambda x: x)(sigmas)
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # length N+1, last = 0

        B = labels.shape[0] if labels is not None else self.num_samples
        if labels is not None:
            labels = labels.to(device=self.device, dtype=dt_f)
        if net_obs is not None:
            net_obs = net_obs.to(device=self.device, dtype=dt_f)
            args = (labels, net_obs)
        else:
            args = (labels,)

        latents = torch.randn((B, self.num_channels, *self.sample_shape), device=self.device, dtype=dt_t)

        x_next = (latents.to(self.dtype_t) * sigmas[0])

        losses = torch.zeros((num_steps, 3))  # for debugging

        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):  # i = 0..N-1
            x_cur = x_next.detach().clone()
            x_cur.requires_grad = True
            # Euler step to t_next
            x_N, dxdt = self.out_and_grad_fun(self.net, x_cur.to(dt_f), torch.full((B,), sigma_cur, device=self.device, dtype=dt_f), *args)
            x_N, dxdt = x_N.to(dt_t), dxdt.to(dt_t)
            
            d_cur = (x_cur - x_N) / sigma_cur
            x_next = x_cur + (sigma_next - sigma_cur) * d_cur
            # Heun (2nd-order) correction unless final step
            if i < num_steps - 1:
                x_N, dxdt = self.out_and_grad_fun(self.net, x_next.to(dt_f), torch.full((B,), sigma_next, device=self.device, dtype=dt_f), *args)
                x_N, dxdt = x_N.to(dt_t), dxdt.to(dt_t)
                d_prime = (x_next - x_N) / sigma_next
                x_next = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)

            # observation loss for DPS sampling
            loss_u = torch.zeros(1, dtype=dt_t, device=self.device)
            if mask_u.sum() > 0:
                loss_u = torch.sqrt(torch.sum((mask_u * (x_N - obs_u)) ** 2))

            # PDE loss for DPS sampling
            lap_u = laplacian(x_N, self.dx)
            alphas = labels[:, -1].to(dt_t).view(-1, 1, 1, 1) if labels is not None else 1.0
            dudt = dxdt
            loss_pde = torch.sqrt(torch.sum((dudt - alphas * lap_u)**2)) / (self.sample_shape[-2] * self.sample_shape[-1])  # normalize by spatial size

            
            if i <= 0.8 * num_steps:
                w_u, w_pde = zeta_u, zeta_pde
            else:
                w_u, w_pde = 0.1 * zeta_u, zeta_pde
            
            loss_comb = w_u * loss_u + w_pde * loss_pde
            grad_x = torch.autograd.grad(loss_comb, x_cur, retain_graph=False)[0]
            x_next = x_next - grad_x
            
            losses[i] = torch.tensor([loss_u.item(), loss_pde.item(), loss_comb.item()])        
        # Return at sigma=0 in fp32
        x = x_next.to(dt_f).detach().cpu()

        losses = losses.detach().cpu().numpy() if return_losses else None
        return x, losses


class JointSampler(Sampler):
    def __init__(
        self,
        net,
        device,
        sample_shape,
        num_channels,
        num_samples,
        loss_fn,
        loss_kwargs,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ):
        super().__init__(
            net,
            device,
            sample_shape,
            num_channels,
            num_samples,
            num_steps,
            sigma_min,
            sigma_max,
            rho,
    )
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs

    

class sampling_context:
    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def __enter__(self):
        self.prev_fp32_prec = torch.backends.cudnn.conv.fp32_precision
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        self.sampler.net.eval()
        self.sampler.net.to(self.sampler.device)
        #return self.sampler
    
    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cudnn.conv.fp32_precision = self.prev_fp32_prec
        self.sampler.net.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        torch.cuda.synchronize()