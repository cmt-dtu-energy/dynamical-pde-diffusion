import torch
import logging
from .sample import laplacian

logger = logging.getLogger(__name__)

def heat_loss(x, dxdt, obs_a, obs_u, mask_a, mask_u, dx, dy, ch_a, labels):
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
    labels : torch.Tensor
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
    alpha = labels.view(x.shape[0], 1, 1, 1)  # Reshape to (B, 1, 1, 1) for broadcasting
    dudt = dxdt[:, ch_a:, :, :]

    a_N, u_N = x[:, :ch_a, :, :], x[:, ch_a:, :, :]
    laplacian_u = laplacian(u_N, dx)

    loss_pde = torch.norm(dudt - alpha * laplacian_u, 2) / (u_N.shape[-1] * u_N.shape[-2])
    loss_obs_a = torch.norm(mask_a * (a_N - obs_a), 2)
    loss_obs_u = torch.norm(mask_u * (u_N - obs_u), 2)

    return loss_pde, loss_obs_a, loss_obs_u


def llg_loss(x, dxdt, obs_a, obs_u, mask_a, mask_u, dx, dy, ch_a, labels):
    """
    Compute the Landau-Lifshitz-Gilbert (LLG) equation loss components.
    
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
    labels : torch.Tensor
        External magnetic field vector of shape (B, 3).

    Returns 
    -------
    loss_pde : torch.Tensor
        PDE loss component.
    loss_obs_a : torch.Tensor
        Observation loss component for the initial condition.
    loss_obs_u : torch.Tensor
        Observation loss component for the solution at time T.
    """    
    ###gamma = 2.211e5  # gyromagnetic ratio in m/(A s)
    ###alpha = 0.1      # damping constant

    ###m = x[:, :ch_a, :, :]  # Magnetization vector (B, 3, H, W)
    ###dm_dt = dxdt[:, :ch_a, :, :]

    ###H_ext = labels.view(x.shape[0], 3, 1, 1)  # Reshape to (B, 3, 1, 1) for broadcasting

    # Compute effective field (assuming only external field for simplicity)
    ###H_eff = H_ext

    # Compute the LLG right-hand side
    ###mxH = torch.cross(m.permute(0, 2, 3, 1), H_eff.permute(0, 2, 3, 1), dim=-1).permute(0, 3, 1, 2)
    ###m_cross_mxH = torch.cross(m.permute(0, 2, 3, 1), mxH.permute(0, 2, 3, 1), dim=-1).permute(0, 3, 1, 2)

    ###llg_rhs = -gamma / (1 + alpha**2) * (mxH + alpha * m_cross_mxH)

    ###loss_pde = torch.norm(dm_dt - llg_rhs, 2) / (m.shape[-1] * m.shape[-2])

    a = x[:, :ch_a, :, :]  # Magnetization vector (B, 3, H, W)
    u = x[:, ch_a:, :, :]  # Unused channels (B, C - 3, H, W)

    loss_pde = torch.norm(torch.norm(u, p=2, dim=1) - 1, p=2) # enforce |m| = 1 constraint
    loss_obs_a = torch.norm(mask_a * (a - obs_a), p=2)
    loss_obs_u = torch.norm(mask_u * (u - obs_u), p=2)

    return loss_pde, loss_obs_a, loss_obs_u