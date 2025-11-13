import os
import sys
import numpy as np
import torch
import logging

from magtense.micromag import MicromagProblem

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

    loss_pde = torch.norm(dudt - alpha * laplacian_u, 2) / (
        u_N.shape[-1] * u_N.shape[-2]
    )
    loss_obs_a = torch.norm(mask_a * (a_N - obs_a), 2)
    loss_obs_u = torch.norm(mask_u * (u_N - obs_u), 2)

    return loss_pde, loss_obs_a, loss_obs_u


def llg_loss(
    x, dxdt, obs_a, obs_u, mask_a, mask_u, dx, dy, ch_a, labels, cuda: bool = True
):
    """
    Compute the Landau-Lifshitz-Gilbert (LLG) equation loss components.

    LLG in MagTense notation:
        dm_dt = - gamma * (m x H_eff) - alpha * m x (m x H_eff)
        H_eff = H_ext + H_demag + H_exch + H_anis

    Choice for muMAG Std problem #4 (https://www.ctcms.nist.gov/~rdm/std4/spec4.html):
        H_anis = 0
        H_ext = (known) constant external field
        H_demag = computed via magnetostatic equations (not included here)
        H_exch = (2 A / (mu0 Ms)) * Laplacian(m)

    Following parameters are fixed for simplicity, can be modified as needed:
        gamma = 2.21e5  # gyromagnetic ratio [m/(A s)]
        alpha = 4.42e3  # damping constant
        A0 = 1.3e-11    # exchange stiffness [J/m]
        Ms = 8e5        # saturation magnetization [A/m]
        K0 = 0.0        # anisotropy constant [J/m^3]

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
    # Magnetization vector (B, 3, H, W)
    m = x[:, ch_a:, :, :]
    # Initial condition vector (B, ch_a, H, W)
    a = x[:, :ch_a, :, :]
    n_magnets = m.shape[-1] * m.shape[-2]
    # Time derivative of Magnetization vector (B, 3, H, W)
    dmdt = dxdt[:, ch_a:, :, :]

    dtype = torch.float32
    res = [16, 4, 1]
    grid_size = [500e-9, 125e-9, 3e-9]
    mu0 = 4e-7 * torch.pi
    t_per_step = 4e-12
    gamma = 2.21e5
    alpha = 4.42e3
    A0 = 1.3e-11
    Ms = 8e5
    K0 = 0.0

    # Reshape to (B, 3, 1, 1) for broadcasting and bring to units [A/m]
    h_ext = labels.view(x.shape[0], 3, 1, 1) / (1000 * mu0)
    h_eff = torch.zeros_like(m)

    # Iterate over batch dimension to compute fields for each sample
    for i in range(m.shape[0]):
        n_magnets = res[0] * res[1] * res[2]

        # Reshape from  (3, H, W) to (n_magnets, 3)
        m_mt = m[i].swapaxes(1, 2).reshape(3, -1).T.numpy()
        problem_dym = MicromagProblem(
            res=res,
            grid_L=grid_size,
            m0=m_mt,
            alpha=alpha,
            gamma=gamma,
            A0=A0,
            Ms=Ms,
            K0=K0,
            usereturnhall=1,
            cuda=cuda,
        )

        def h_ext_fct(t) -> np.ndarray:
            return np.expand_dims(t > -1, axis=1) * h_ext[i, :, 0, 0].numpy()

        devnull = open("/dev/null", "w")
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        h_e, _, h_d, h_a = problem_dym.run_simulation(
            t_end=t_per_step * 9,
            nt=10,  # Minimum number of steps for rksuite of MagTense to work properly
            fct_h_ext=h_ext_fct,
            nt_h_ext=100,
        )[3:7]
        os.dup2(oldstdout_fno, 1)

        # Reshape back to (3, H, W) and negate fields as MagTense returns -H
        h_exch = torch.tensor(
            -h_e[1, :, 0].copy().T.reshape(3, res[1], res[0]).swapaxes(1, 2),
            dtype=dtype,
        )
        h_demag = torch.tensor(
            -h_d[1, :, 0].copy().T.reshape(3, res[1], res[0]).swapaxes(1, 2),
            dtype=dtype,
        )
        h_anis = torch.tensor(
            -h_a[1, :, 0].copy().T.reshape(3, res[1], res[0]).swapaxes(1, 2),
            dtype=dtype,
        )
        h_eff[i] = h_ext[i] + h_exch + h_demag + h_anis

    # Compute the LLG right-hand side
    mxH = torch.cross(m, h_eff, dim=1)
    m_cross_mxH = torch.cross(m, mxH, dim=1)
    llg_rhs = -gamma * mxH - alpha * m_cross_mxH

    loss_pde = torch.norm(dmdt - llg_rhs * t_per_step, p=2, dim=1) / n_magnets
    # enforce |m| = 1 constraint
    loss_norm = torch.norm(m, p=2, dim=1) - 1

    loss_obs_a = torch.norm(mask_a * (a - obs_a), p=2)
    loss_obs_u = torch.norm(mask_u * (m - obs_u), p=2)

    return loss_pde, loss_obs_a, loss_obs_u
