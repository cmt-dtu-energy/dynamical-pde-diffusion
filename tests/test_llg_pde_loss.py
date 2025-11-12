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
"""

import os
import h5py
import numpy as np
import torch
import sys

from pathlib import Path
from magtense.magstatics import Tiles, run_simulation

from diffusion_pde.sampling.sample import laplacian

mu0 = 4e-7 * torch.pi
t_per_step = 4e-12
gamma = 2.21e5
alpha = 4.42e3
A0 = 1.3e-11
Ms = 8e5
K0 = 0.0

db = h5py.File(Path(__file__).parent / ".." / "data" / "4_500_16_4.h5", "r")
# shape: (n_samples, t_steps, 3, res_x, res_y)
m_sequences = db["sequence"][:]
# shape: (n_samples, 3)
h_ext_n = db["field"][:]
grid_size = db.attrs["grid_size"]
res = db.attrs["res"]
dx = grid_size[0] / res[0]
dy = grid_size[1] / res[1]
dz = grid_size[2] / res[2]
n_magnets = res[0] * res[1] * res[2]
t_steps = int(db.attrs["t_steps"])
db.close()


def llg_loss_individual(seed: int = 0):
    torch.manual_seed(seed)
    n = torch.randint(0, m_sequences.shape[0], (1,)).item()
    t = torch.randint(0, t_steps - 1, (1,)).item()
    m = torch.tensor(m_sequences[n, t], dtype=torch.float32)  # (3, H, W)
    dmdt = torch.tensor(m_sequences[n, t + 1], dtype=torch.float32) - m.clone()

    # Reshape to (3, 1, 1) for broadcasting
    h_ext = torch.tensor(h_ext_n[n], dtype=torch.float32).view(3, 1, 1)
    h_eff = torch.zeros_like(m)
    loc = np.meshgrid(
        np.linspace(0, grid_size[0], res[0]),
        np.linspace(0, grid_size[1], res[1]),
        np.linspace(0, grid_size[2], res[2]),
        indexing="ij",
    )
    loc_np = np.stack(loc, axis=-1).reshape(-1, 3)

    ## Option 1: Calculation of individual field components
    # Exchange field
    laplacian_m = laplacian(m.unsqueeze(1), dx)
    h_exch = (2 * A0 / (mu0 * Ms)) * laplacian_m.squeeze(1)

    # Anisotropy field
    h_anis = torch.zeros_like(m)

    # Demagnetisation field
    tiles = Tiles(
        n=n_magnets,
        M_rem=Ms,
        easy_axis=m.reshape(3, -1).T.numpy(),
        tile_type=2,
        size=[dx, dy, dz],
        offset=loc_np,
    )
    devnull = open("/dev/null", "w")
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    h_out = run_simulation(tiles, loc_np)[1]
    os.dup2(oldstdout_fno, 1)
    h_demag = (
        torch.tensor(h_out.reshape(res[0], res[1], 3), dtype=torch.float32).permute(
            2, 0, 1
        )
        * mu0
    )

    # Compute effective field
    h_eff = h_ext + h_exch + h_demag + h_anis

    # Compute the LLG right-hand side
    mxH = torch.cross(m, h_eff, dim=0)
    m_cross_mxH = torch.cross(m, mxH, dim=0)
    llg_rhs = -gamma * mxH - alpha * m_cross_mxH

    loss_pde = torch.norm(dmdt - llg_rhs * t_per_step, dim=0) / n_magnets

    print("Largest errors: ", torch.sort(torch.abs(loss_pde.reshape(-1)))[0][-5:])
    # print("Norm m: ", torch.norm(m, dim=0).reshape(-1)[0])
    assert torch.all(torch.abs(loss_pde.reshape(-1)) < 2e-4)


# def llg_loss_mt(x, dxdt, labels, ch_a):


#         ### Option 2: Get solution directly from MagTense
#         problem_dym = magtense.MicromagProblem(
#             res=res,
#             grid_L=grid_size,
#             m0=m[i],
#             alpha=alpha,
#             gamma=gamma,
#             grid_pts=None,
#             grid_abc=None,
#             grid_type="uniform",
#             exch_rows=None,
#             exch_col=None,
#             exch_val=None,
#             exch_nval=1,
#             exch_nrow=1,
#             exch_ncols=1,
#             passexch=0,
#             cuda=True,
#             cvode=False,
#         )

#         k = np.moveaxis(m.reshape(res[1], res[0], res[2], 3).swapaxes(0, 1), -1, 0)[
#             :, :, :, 0
#         ]

#         def h_ext_fct(t) -> np.ndarray:
#             return np.expand_dims(t > -1, axis=1) * (
#                 H_ext[i].cpu().numpy() / 1000 / mu0
#             )

#         H_exch, _, H_demag, H_anis = problem_dym.run_simulation(
#             t_end=t_per_step,
#             nt=1,
#             fct_h_ext=h_ext_fct,
#             nt_h_ext=1,
#         )[3:7]

#         H_eff[i] = (
#             H_ext[i]
#             + torch.tensor(H_exch.copy())
#             + torch.tensor(H_demag.copy())
#             + torch.tensor(H_anis.copy())
#         )

if __name__ == "__main__":
    llg_loss_individual()
