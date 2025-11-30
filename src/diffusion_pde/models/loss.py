import torch
import logging
from abc import ABC, abstractmethod
from diffusion_pde.sampling import X_and_dXdt_fd, laplacian
logger = logging.getLogger(__name__)

class Loss(ABC):
    @abstractmethod
    def __call__(self, net, x, labels, run=None, **kwargs):
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

    def __call__(self, net, x, labels, run=None, global_step=None, **kwargs):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x, device=x.device) * sigma
        D_yn = net(x + n, sigma.flatten(), labels, **kwargs)
        loss = weight * ((D_yn - x) ** 2)

        if run is not None:
            run.log({"Loss/train/batch/EDM": loss.mean().item()}, step=global_step)
        if self.reduce_method == "mean":
            return loss.mean(dim=(1,2,3))
        elif self.reduce_method == "sum":
            return loss.sum(dim=(1,2,3))
        


class EDMHeatLoss(Loss):
    """
    Loss function for EDM training with heat equation PDE loss
    This loss follows the Mean Estimation (ME) variant of Physics Informed Diffusion Models (PIDM)
    from the paper:
    "Physics Informed Diffusion Models" : https://arxiv.org/pdf/2403.14404
    
    The PDE loss is simply evaluated on the denoised estimate D_yn
    """
    def __init__(self,
        dx,
        pde_loss_coeff=1.0, 
        method="joint",
        residual_estimation="ME",   # 'ME' or 'SE'
        P_mean=-1.2, 
        P_std=1.2, 
        sigma_data=0.5, 
        reduce_method="mean",
        sigma_min=0.01,
        rho=7.0,
        steps=2,
        ):
        assert method in ["joint", "forward"], "method must be either 'joint' or 'forward'"
        assert residual_estimation in ["ME", "SE"], "residual_estimation must be either 'ME' or 'SE'"

        self.dx = dx
        self.pde_loss_coeff = pde_loss_coeff
        self.residual_estimation = residual_estimation
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.reduce_method = reduce_method
        self.sigma_min = sigma_min
        self.rho = rho
        self.steps = steps
        self.ch_a = 1 if method == "joint" else 0

    def two_step_sample(self, net, x, sigma_max, labels, **net_kwargs):
        """
        Short 2-step EDM-style sampler used during training to get x0* and dxdt
        for the PDE loss.

        x         : starting noisy input (B, C, H, W), e.g. x0 + sigma * eps
        sigma_max : per-batch initial sigma, shape (B,) or (B, 1, 1, 1)
        labels    : conditioning (whatever your out_and_grad_fun needs)
        sigma_min : minimum sigma to integrate to (close-ish to data)
        num_steps : number of ODE steps between sigma_max and sigma_min (2 in paper)
        """

        device = x.device
        B = x.shape[0]

        # Ensure sigma_max is (B,) on correct device
        sigma_max = sigma_max.view(B)

        # Build per-sample sigma schedule, as in EDM: sigmas ∈ [sigma_max, ..., sigma_min, 0]
        sigma_min = torch.tensor(float(self.sigma_min), device=device, dtype=torch.float32)
        # step_idx: 0, 1, ..., num_steps-1
        step_idx = torch.arange(self.steps + 1, dtype=torch.float32, device=device)  # (num_steps,)

        # For each sample i in batch, create its own schedule
        # sigmas[i, :] has length num_steps+1: from sigma_max[i] to sigma_min, then we append 0
        sigmas_list = []
        for i in range(B):
            s_max = sigma_max[i]
            s_seq = (
                s_max**(1.0 / self.rho)
                + step_idx / self.steps * (sigma_min**(1.0 / self.rho) - s_max**(1.0 / self.rho))
            )**self.rho  # (num_steps,)
            sigmas_list.append(s_seq)

        # Shape: (B, num_steps+1)
        sigmas = torch.stack(sigmas_list, dim=0)  # (B, N+1)

        # We'll iterate over sigma pairs along axis 1: (sigma_cur, sigma_next)
        # and carry a batch x_next.
        x_next = x

        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas.T[:-1], sigmas.T[1:])):  # i = 0..num_steps-1

            # x_cur is a fresh variable we can differentiate w.r.t. inside out_and_grad_fun
            x_cur = x_next
            # Call your helper: should return (x_N, dxdt) with full batch
            # sigma_cur needs to be float32 for the net

            x_N = net(x_cur, sigma_cur.flatten(), labels, **net_kwargs)

            # First-order EDM PF-ODE velocity: d = (x - x_N) / sigma
            sigma_cur_b = sigma_cur.view(B, 1, 1, 1)
            d_cur = (x_cur - x_N) / sigma_cur_b

            sigma_next_b = sigma_next.view(B, 1, 1, 1)
            x_next = x_cur + (sigma_next_b - sigma_cur_b) * d_cur

        return x_next


    def __call__(self, net, x, labels, run=None, global_step=None, **kwargs):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x, device=x.device) * sigma
        D_yn, dxdt = X_and_dXdt_fd(net, x + n, sigma.flatten(), labels, **kwargs, no_grad=False)
        dxdt = dxdt.detach()
        dxdt = dxdt[:, self.ch_a:, ...]  # select channel for PDE loss
        #D_yn = net(x + n, sigma.flatten(), *args, **kwargs)
        edm_loss = weight * ((D_yn - x) ** 2)

        if self.residual_estimation == "ME":
            x_0star = D_yn
        elif self.residual_estimation == "SE":
            x_0star = self.two_step_sample(net, D_yn, sigma, labels, **kwargs)

        pde_loss = (dxdt - labels[:, 1].view(-1, 1, 1, 1) * laplacian(x_0star[:, self.ch_a:, ...], self.dx)) ** 2 / (x.shape[-2] * x.shape[-1])
        if self.reduce_method == "mean":
            edm_loss = edm_loss.mean(dim=(1,2,3))
            pde_loss = pde_loss.mean(dim=(1,2,3)) * self.pde_loss_coeff / (sigma ** 2)
        elif self.reduce_method == "sum":
            edm_loss = edm_loss.sum(dim=(1,2,3))
            pde_loss = pde_loss.sum(dim=(1,2,3)) * self.pde_loss_coeff / (sigma ** 2)

        loss = edm_loss + pde_loss

        if run is not None:
            run.log({
                "Loss/train/batch/EDM": edm_loss.mean().item(),
                "Loss/train/batch/PDE": pde_loss.mean().item(),
                "Loss/train/batch/Total": loss.mean().item(),
            }, step=global_step)
        
        return loss
    