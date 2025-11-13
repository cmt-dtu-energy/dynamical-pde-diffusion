"""
Compute the Landau-Lifshitz-Gilbert (LLG) equation loss components.

LLG in MagTense notation:
    dm_dt = - gamma * (m x H_eff) - alpha * m x (m x H_eff)
    H_eff = H_ext + H_demag + H_exch + H_anis

Choice for muMAG Std problem #4 (https://www.ctcms.nist.gov/~rdm/std4/spec4.html):
    H_anis = 0
    H_ext = (known) constant external field
    H_demag = computed via magnetostatic equations (not included here)
    H_exch = (2 * A0 / (mu0 * Ms)) * Laplacian(m)

Following parameters are fixed for simplicity, can be modified as needed:
    gamma = 2.21e5  # gyromagnetic ratio [m/(A s)]
    alpha = 4.42e3  # damping constant
    A0 = 1.3e-11    # exchange stiffness [J/m]
    Ms = 8e5        # saturation magnetization [A/m]
    K0 = 0.0        # anisotropy constant [J/m^3]
"""

import os
import h5py
import numpy as np
import torch
import sys

from pathlib import Path
from magtense.magstatics import Tiles, run_simulation
from magtense.micromag import MicromagProblem
from magtense.utils import plot_M_thin_film

from diffusion_pde.sampling.sample import laplacian

dtype = torch.float32
mu0 = 4e-7 * torch.pi
gamma = 2.21e5
alpha = 4.42e3
A0 = 1.3e-11
Ms = 8e5
K0 = 0.0

figpath = Path(__file__).parent / ".." / "figs"
db = h5py.File(Path(__file__).parent / ".." / "data" / "4_500_16_4.h5", "r")
# shape: (n_samples, t_steps, 3, res_x, res_y)
m_sequences = db["sequence"][:]
# shape: (n_samples, 3)
h_ext_n = db["field"][:]
grid_size = db.attrs["grid_size"]
res = db.attrs["res"]
t_steps = int(db.attrs["t_steps"])
t_per_step = float(db.attrs["t_per_step"])
db.close()


def llg_loss_individual(seed: int = 0, n_t: int = 1, plot: bool = False):
    torch.manual_seed(seed)
    dx = grid_size[0] / res[0]
    dy = grid_size[1] / res[1]
    dz = grid_size[2] / res[2]
    n_magnets = res[0] * res[1] * res[2]

    n = torch.randint(0, m_sequences.shape[0], (1,)).item()
    t = torch.randint(0, t_steps - n_t, (1,)).item()
    m = torch.tensor(m_sequences[n, t], dtype=dtype)  # (3, H, W)
    m_t = torch.tensor(m_sequences[n, t + n_t], dtype=dtype)
    dmdt = m_t.clone() - m.clone()

    # Unit [A/m] and reshape to (3, 1, 1) for broadcasting
    h_ext = torch.tensor(h_ext_n[n], dtype=dtype).view(3, 1, 1) / (1000 * mu0)
    h_eff = torch.zeros_like(m)
    loc = np.meshgrid(
        np.linspace(0, grid_size[0], res[0]),
        np.linspace(0, grid_size[1], res[1]),
        np.linspace(0, grid_size[2], res[2]),
        indexing="ij",
    )
    loc_np = np.stack(loc, axis=-1).reshape(-1, 3)

    ### Option 1: Calculation of individual field components
    # Exchange field
    laplacian_m = torch.squeeze(laplacian(m.unsqueeze(1), dx), dim=1)
    # Unit [A/m]
    h_exch = (2 * A0 / (mu0 * Ms)) * laplacian_m

    # Anisotropy field
    h_anis = torch.zeros_like(m)

    # Demagnetisation field
    m_magstat = m.reshape(3, -1).T.numpy()
    tiles = Tiles(
        n=n_magnets,
        M_rem=Ms,
        easy_axis=m_magstat,
        tile_type=2,
        size=[dx, dy, dz],
        offset=loc_np,
    )
    devnull = open("/dev/null", "w")
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    # MagTense returns H in unit [A/m]
    h_out = run_simulation(tiles, loc_np)[1]
    os.dup2(oldstdout_fno, 1)
    h_demag = torch.tensor(h_out.reshape(res[0], res[1], 3), dtype=dtype).permute(
        2, 0, 1
    )

    # Compute effective field
    h_eff = h_ext + h_anis + h_demag + h_exch

    # Compute the LLG right-hand side
    mxH = torch.cross(m, h_eff, dim=0)
    m_cross_mxH = torch.cross(m, mxH, dim=0)
    llg_rhs = -gamma * mxH - alpha * m_cross_mxH

    loss_pde = torch.norm(dmdt - llg_rhs * t_per_step * n_t, dim=0) / n_magnets

    if plot:
        m_t_comp = llg_rhs * t_per_step * n_t + m.clone()
        plot_M_thin_film(
            m.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            f"opt1_{seed}_m",
            figpath=figpath,
        )
        plot_M_thin_film(
            (dmdt - llg_rhs * t_per_step * n_t).swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt1_{seed}_loss_pde",
            figpath=figpath,
        )
        plot_M_thin_film(
            m_t_comp.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt1_{seed}_m_t",
            figpath=figpath,
        )
        plot_M_thin_film(
            m_t.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt1_{seed}_m_t_valid",
            figpath=figpath,
        )
        plot_M_thin_film(
            dmdt.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt1_{seed}_dmdt",
            figpath=figpath,
        )

        plot_M_thin_film(
            h_exch.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt1_{seed}_h_exch",
            figpath=figpath,
        )
        plot_M_thin_film(
            h_demag.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt1_{seed}_h_demag",
            figpath=figpath,
        )

    print(
        "[Opt1] Largest errors: ", torch.sort(torch.abs(loss_pde.reshape(-1)))[0][-5:]
    )
    print("[Opt1] Mean error: ", torch.mean(loss_pde))

    # assert torch.all(torch.abs(loss_pde.reshape(-1)) < 1e-4)


def llg_loss_mt(seed: int = 0, n_t: int = 1, cuda: bool = True, plot: bool = False):
    torch.manual_seed(seed)
    n_magnets = res[0] * res[1] * res[2]

    n = torch.randint(0, m_sequences.shape[0], (1,)).item()
    t = torch.randint(0, t_steps - n_t, (1,)).item()
    m = torch.tensor(m_sequences[n, t], dtype=dtype)  # (3, H, W)
    m_t = torch.tensor(m_sequences[n, t + n_t], dtype=dtype)
    dmdt = m_t.clone() - m.clone()

    h_ext = torch.tensor(h_ext_n[n], dtype=dtype).view(3, 1, 1) / (1000 * mu0)
    h_eff = torch.zeros_like(m)

    ### Option 2: Get solution directly from MagTense
    # Reshape from  (3, H, W) to (n_magnets, 3)
    m_mt = m.swapaxes(1, 2).reshape(3, -1).T.numpy()
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
        return np.expand_dims(t > -1, axis=1) * h_ext[:, 0, 0].numpy()

    devnull = open("/dev/null", "w")
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    _, m_out, _, h_e, _, h_d, h_a = problem_dym.run_simulation(
        t_end=t_per_step * n_t * 10,
        nt=10,  # Minimum number of steps for rksuite of MagTense to work properly
        fct_h_ext=h_ext_fct,
        nt_h_ext=100,
    )[:7]
    os.dup2(oldstdout_fno, 1)

    # Reshape back to (3, H, W) and negate fields as MagTense returns -H
    # m_t_valid = torch.tensor(
    #     m_out[1, :, 0].copy().T.reshape(3, res[1], res[0]).swapaxes(1, 2),
    #     dtype=dtype,
    # )
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

    h_eff = h_ext + h_exch + h_demag + h_anis

    # Compute the LLG right-hand side
    mxH = torch.cross(m, h_eff, dim=0)
    m_cross_mxH = torch.cross(m, mxH, dim=0)
    llg_rhs = -gamma * mxH - alpha * m_cross_mxH

    loss_pde = torch.norm(dmdt - llg_rhs * t_per_step, dim=0) / n_magnets
    # loss_m = torch.norm(m_t - m_t_valid, dim=0) / n_magnets

    if plot:
        m_t_comp = llg_rhs * t_per_step + m.clone()
        plot_M_thin_film(
            m.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            f"opt2_{seed}_m",
            figpath=figpath,
        )
        plot_M_thin_film(
            (dmdt - llg_rhs * t_per_step).swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt2_{seed}_loss_pde",
            figpath=figpath,
        )
        plot_M_thin_film(
            m_t_comp.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt2_{seed}_m_t",
            figpath=figpath,
        )
        plot_M_thin_film(
            m_t.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt2_{seed}_m_t_valid",
            figpath=figpath,
        )
        plot_M_thin_film(
            m_t.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt2_{seed}_m_mt_t",
            figpath=figpath,
        )
        plot_M_thin_film(
            dmdt.swapaxes(1, 2).reshape(3, -1).T.numpy(),
            res,
            title=f"opt2_{seed}_dmdt",
            figpath=figpath,
        )
        plot_M_thin_film(
            -h_e[1, :, 0].copy(),
            res,
            title=f"opt2_{seed}_h_exch",
            figpath=figpath,
        )
        plot_M_thin_film(
            -h_d[1, :, 0].copy(),
            res,
            title=f"opt2_{seed}_h_demag",
            figpath=figpath,
        )

    print(
        "[Opt2] Largest errors: ", torch.sort(torch.abs(loss_pde.reshape(-1)))[0][-5:]
    )
    print("[Opt2] Mean error: ", torch.mean(loss_pde))
    # print(
    #     "[Opt2] Largest errors (m_t): ",
    #     torch.sort(torch.abs(loss_m.reshape(-1)))[0][-5:],
    # )
    # assert torch.all(torch.abs(loss_m.reshape(-1)) < 5e-5)


if __name__ == "__main__":
    for seed in range(5):
        print(f"--- Seed {seed} ---")
        llg_loss_individual(seed=seed)
        llg_loss_mt(seed=seed, cuda=True)
