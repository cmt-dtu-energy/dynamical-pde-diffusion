import torch
from .sample import laplacian

def heat_loss(a_N, u_N, dudt, obs_a, obs_u, mask_a, mask_u, dx):
    # Compute losses

    laplacian_u = laplacian(u_N, dx)

    loss_pde = torch.norm(dudt - laplacian_u, 2) / (u_N.shape[-1] * u_N.shape[-2])
    loss_obs_a = torch.norm(mask_a * (a_N - obs_a), 2)
    loss_obs_u = torch.norm(mask_u * (u_N - obs_u), 2)

    return loss_pde, loss_obs_a, loss_obs_u