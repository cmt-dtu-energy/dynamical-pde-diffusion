import torch
from .sample import laplacian

def heat_loss(x, dxdt, obs_a, obs_u, mask_a, mask_u, dx, dy, ch_a, label):
    """
    Compute the heat equation loss components.
    
    Parameters
    ----------
    x : torch.Tensor
        Current state tensor of shape (B, C, H, W), where C = ch_a + ch_u.
    dxdt : torch.Tensor
        Time derivative of x, tensor of shape (B, C, H, W).
    obs_a : torch.Tensor
        Observations of the initial condition, tensor of shape (B, ch_a, H, W).
    obs_u : torch.Tensor
        Observations of the solution at time T, tensor of shape (B, ch_u, H, W).
    mask_a : torch.Tensor
        Binary mask for obs_a, tensor of shape (B, ch_a, H, W).
    mask_u : torch.Tensor
        Binary mask for obs_u, tensor of shape (B, ch_u, H, W).
    dx : float
        Spatial grid size in x-direction.
    dy : float
        Spatial grid size in y-direction.
    ch_a : int
        Number of channels for the initial condition.
    label : torch.Tensor
        Diffusion coefficient.

    Returns
    -------
    loss_pde : torch.Tensor
        PDE loss component.
    loss_obs_a : torch.Tensor
        Observation loss component for the initial condition.
    loss_obs_u : torch.Tensor
        Observation loss component for the solution at time T.
    """
    alpha = float(label.squeeze())
    dudt = dxdt[:, ch_a:, :, :]

    a_N, u_N = x[:, :ch_a, :, :], x[:, ch_a:, :, :]
    laplacian_u = laplacian(u_N, dx)

    loss_pde = torch.norm(dudt - alpha * laplacian_u, 2) / (u_N.shape[-1] * u_N.shape[-2])
    loss_obs_a = torch.norm(mask_a * (a_N - obs_a), 2)
    loss_obs_u = torch.norm(mask_u * (u_N - obs_u), 2)

    return loss_pde, loss_obs_a, loss_obs_u