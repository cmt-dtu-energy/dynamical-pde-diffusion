import math
import torch
import numpy as np
from diffusion_pde.utils import get_repo_root
from diffusion_pde.pdes import save_data
from diffusion_pde.pdes.heat import make_grid, dirichlet_sine_basis, sine2d_forward, sine2d_inverse, linear_bc_field, random_gaussian_blobs


@torch.no_grad()
def generate_heat_no_cond_batched(
    B: int,
    T: int,                     # final time
    S: int,                     # spatial grid size
    X: torch.Tensor,            # (S, S) x grid points   
    Y: torch.Tensor,            # (S, S) y grid points
    S_int: torch.Tensor,        # (S-2, S-2) interior DST matrix
    lam2d_int: torch.Tensor,    # (S-2, S-2) interior eigenvalues
    device: str = "cuda",       
    dtype: torch.dtype = torch.float32,
    ic_seed: int | None = None,
    n_blobs: tuple[int, int] = (1, 3)
):
    device = torch.device(device)
    a = torch.empty(B, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b = torch.empty(B, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    c = torch.empty(B, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    # Lift for BCs
    w = linear_bc_field(a, b, c, X, Y)  # (B,S,S)

    # Initial condition (random blobs), then force boundary to w
    u0_raw = random_gaussian_blobs(B, S, X, Y, device=device, dtype=dtype, seed=ic_seed, n_blobs=n_blobs) # (B,S,S)
    u0 = u0_raw.clone()
    u0[:,  0, :] = w[:,  0, :]
    u0[:, -1, :] = w[:, -1, :]
    u0[:, :,  0] = w[:, :,  0]
    u0[:, :, -1] = w[:, :, -1]

    # Homogeneous interior residual
    v0 = u0 - w                      # (B,S,S)
    v0_in = v0[:, 1:-1, 1:-1]        # (B,S-2,S-2)

    # Forward interior DST
    V_hat = sine2d_forward(v0_in, S_int)  # (B,S-2,S-2)

    # Prepare time series buffer
    u_ts = torch.empty((B, S, S, 2), device=device, dtype=dtype)
    u_ts[..., 0] = u0

    # Broadcastables
    lam2d = lam2d_int.to(device=device, dtype=dtype)            # (S-2,S-2)

    V_hat_t = V_hat.clone()
    decay = torch.exp(-lam2d * T)    # (B,S-2,S-2) via broadcast
    V_hat_t = V_hat_t * decay
    v_in = sine2d_inverse(V_hat_t, S_int)            # (B,S-2,S-2)

        # re-embed interior & add lift
    v_full = torch.zeros((B, S, S), device=device, dtype=dtype)
    v_full[:, 1:-1, 1:-1] = v_in
    u_t = v_full + w
    u_ts[..., 1] = u_t

    return u_ts


def generate_heat_no_cond(
    N: int,
    T: int,                     # final time
    S: int,                     # spatial grid size
    Lx: float = 1.0,            # x domain size
    Ly: float = 1.0,            # y domain size
    batch_size: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    ic_seed: int | None = None,
    n_blobs: tuple[int, int] = (1, 3),
):
    X, Y = make_grid(S, Lx, Ly, device=device, dtype=dtype)   # (S,S), (S,S)
    N_int = S - 2
    S_int, lam2d_int = dirichlet_sine_basis(N_int, Lx, Ly, device=device, dtype=dtype)

    u_data = torch.empty((N, S, S, 2), dtype=dtype)

    start = 0
    while start < N:
        b_size = min(batch_size, N - start)

        u_batch = generate_heat_no_cond_batched(
            B=b_size,
            T=T,
            S=S,
            X=X,
            Y=Y,
            S_int=S_int,
            lam2d_int=lam2d_int,
            device=device,
            dtype=dtype,
            ic_seed=(ic_seed + start if ic_seed is not None else None),
            n_blobs=n_blobs,
        )   # (B,S,S,2)

        end = start + b_size
        u_data[start:end, ...] = u_batch.detach().cpu()
        start = end
    u_data = u_data.unsqueeze(1)   # (N,1,S,S,2)
    return u_data.numpy()


def main():
    print("Generating heat equation dataset without conditioning...")
    N = 500          # number of samples
    T = 0.005           # final time
    S = 64            # spatial grid size
    Lx = 1.0          # x domain size
    Ly = 1.0          # y domain size
    batch_size = 64
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    ic_seed = 42
    n_blobs = (4, 8)

    u_data = generate_heat_no_cond(
        N=N,
        T=T,
        S=S,
        Lx=Lx,
        Ly=Ly,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        ic_seed=ic_seed,
        n_blobs=n_blobs,
    )   # (N,S,S,2)

    save_path = get_repo_root() / "data" / "heat_no_cond_test2.hdf5"
    save_data(
        filepath=str(save_path),
        A=u_data[..., 0],
        U=u_data,
        labels=None,
        t_steps=np.array([0.0, T], dtype=np.float32),
        T=T,
        dx=Lx / (S - 1),
        dy=Ly / (S - 1),
        S=S,
        Lx=Lx,
        Ly=Ly,
        n_blobs=n_blobs,
        notes="Heat equation dataset without conditioning: u_t = u_xx + u_yy, Dirichlet BCs with linear lift.",
    )
    print(f"Saved dataset to {save_path}")


if __name__ == "__main__":
    main()