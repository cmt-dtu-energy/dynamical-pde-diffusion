import math
import torch
import numpy as np
from pathlib import Path
from scipy.io import savemat
from diffusion_pde.utils import get_repo_root

# Code for generating 2D heat equation data with linear Dirichlet BCs using sine basis transforms.
# The method is a batched sine-pseudo-spectral method with lifting to handle non-homogeneous BCs.

# -------------------------
# Utilities: grid & sine basis
# -------------------------
def make_grid(N: int = 64, Lx: float = 1.0, Ly: float = 1.0, *, device=None, dtype=None):
    x = torch.linspace(0., Lx, N, device=device, dtype=dtype)
    y = torch.linspace(0., Ly, N, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')   # (N,N)
    return X, Y

def dirichlet_sine_basis(N: int = 64, Lx: float = 1.0, Ly: float = 1.0, *, device=None, dtype=None):
    """
    Orthonormal sine basis matrix S (N x N) for Dirichlet on a uniform grid of N interior samples per axis,
    with eigenvalues lam_x[m], lam_y[n] for m,n = 1..N.
    Inverse is S.T (S is orthonormal).
    """
    # Indices 1..N
    n = torch.arange(1, N+1, device=device, dtype=dtype).view(1, -1)  # (1,N)
    j = torch.arange(1, N+1, device=device, dtype=dtype).view(-1, 1)  # (N,1)
    # Orthonormal DST-II-like matrix: S[j,n] = sqrt(2/(N+1)) * sin(pi * j * n / (N+1))
    S = torch.sqrt(torch.tensor(2.0/(N+1), device=device, dtype=dtype)) * torch.sin(math.pi * j * n / (N+1))   # (N,N)

    # Eigenvalues for Laplacian with Dirichlet BCs
    kx = math.pi * n.squeeze(0) / Lx   # (N,)
    ky = math.pi * n.squeeze(0) / Ly   # (N,)
    lam_x = kx**2
    lam_y = ky**2
    # 2D eigenvalues: lam_2d[i,j] = lam_x[j] + lam_y[i]
    lam2d = lam_y.view(N,1) + lam_x.view(1,N)  # (N,N)
    return S, lam2d

def sine2d_forward(U: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    2D forward transform: U_hat = S @ U @ S^T, batched over leading dims.
    U: (..., N, N), S: (N, N)
    returns (..., N, N)
    """
    # Left multiply
    tmp = torch.matmul(S, U)               # (..., N, N)
    # Right multiply by S^T
    U_hat = torch.matmul(tmp, S.t())
    return U_hat

def sine2d_inverse(U_hat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    Inverse transform (since S is orthonormal): U = S^T @ U_hat @ S
    """
    tmp = torch.matmul(U_hat, S)           # (..., N, N)
    U = torch.matmul(S.t(), tmp)
    return U

# -------------------------
# Linear BC lift: w(x,y) = a + b x + c y
# -------------------------
def linear_bc_field(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, X: torch.Tensor, Y: torch.Tensor):
    """
    a,b,c: (B,)  |  X,Y: (N,N)  -> returns w: (B,N,N)
    """
    # Broadcast (B,1,1) * (N,N)
    return a.view(-1,1,1) + b.view(-1,1,1)*X + c.view(-1,1,1)*Y

# -------------------------
# Random Gaussian blob IC (batched)
# -------------------------
def random_gaussian_blobs(B: int, N: int, X: torch.Tensor, Y: torch.Tensor, *,
                          n_blobs=(1,3), amp_range=(0.5, 1.0),
                          sigma_range=(0.03, 0.15), device=None, dtype=None, seed=None):
    """
    Returns (B,N,N) blobs. Positions/sigmas/amps are randomized per-sample and per-blob.
    Boundary is NOT clamped here (we'll clamp to BC after).
    """
    if seed is not None:
        torch.manual_seed(seed)
    BLOB_MIN, BLOB_MAX = n_blobs
    BLOB_MIN = int(BLOB_MIN); BLOB_MAX = int(BLOB_MAX)
    X = X.to(device=device, dtype=dtype)
    Y = Y.to(device=device, dtype=dtype)

    out = torch.zeros((B, N, N), device=device, dtype=dtype)
    for b in range(B):
        k = torch.randint(BLOB_MIN, BLOB_MAX+1, (1,), device=device).item()
        u = torch.zeros((N,N), device=device, dtype=dtype)
        for _ in range(k):
            cx = torch.rand((), device=device, dtype=dtype)
            cy = torch.rand((), device=device, dtype=dtype)
            sx = torch.empty((), device=device, dtype=dtype).uniform_(sigma_range[0], sigma_range[1])
            sy = torch.empty((), device=device, dtype=dtype).uniform_(sigma_range[0], sigma_range[1])
            amp = torch.empty((), device=device, dtype=dtype).uniform_(amp_range[0], amp_range[1])
            ga = amp * torch.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2)))
            # Random sign for variety
            if torch.rand(()) < 0.5:
                ga = -ga
            u = u + ga
        out[b] = u
    return out

# -------------------------
# Heat evolution with linear Dirichlet BCs via lifting + sine basis
# -------------------------
@torch.no_grad()
def heat_timeseries_linear_bc(
    B: int,                 # batch size    
    steps: int,             # number of time steps to simulate
    dt: float,              # time step size
    alpha: torch.Tensor,    # (B,) diffusivities
    a: torch.Tensor,        # (B,) BC coeff for constant term
    b: torch.Tensor,        # (B,) BC coeff for x term
    c: torch.Tensor,        # (B,) BC coeff for y term
    S: int,                 # grid size (S x S)
    X: torch.Tensor,
    Y: torch.Tensor,
    S_dst: torch.Tensor,
    lam2d: torch.Tensor,
    device: str = "cpu",    # torch device
    dtype: torch.dtype = torch.float32, # torch dtype
    ic_seed: int | None = None,         # random seed for IC generation (None = random each call
):
    """
    Returns:
      u_ts: (B, steps+1, N, N) time series satisfying u|_{∂Ω} = a + b x + c y
      meta: dict with X,Y,S,lam2d if you want to reuse across calls
    """
    device = torch.device(device)
    # Grid and basis (reusable)
    #X, Y = make_grid(S, Lx, Ly, device=device, dtype=dtype)     # (S,S)
    #s, lam2d = dirichlet_sine_basis(S, Lx, Ly, device=device, dtype=dtype)  # (S,S), (S,S)

    # Linear BC field per batch
    w = linear_bc_field(a, b, c, X, Y)                          # (B,S,S)

    # Random Gaussian blob IC per batch
    u0_raw = random_gaussian_blobs(B, S, X, Y, device=device, dtype=dtype, seed=ic_seed)
    # Clamp boundary to BC so that v0 = u0 - w is zero on the boundary (Dirichlet homogeneous)
    u0 = u0_raw.clone()
    u0[:,  0, :] = w[:,  0, :]
    u0[:, -1, :] = w[:, -1, :]
    u0[:, :,  0] = w[:, :,  0]
    u0[:, :, -1] = w[:, :, -1]

    # Residual with homogeneous Dirichlet
    v0 = u0 - w                                              # (B,S,S)

    # Transform to sine basis
    V_hat = sine2d_forward(v0, S_dst)                              # (B,S,S)

    # Per-sample exponential decay per mode
    # decay_k = exp(-alpha_b * lam2d * dt)
    # We'll apply it step-by-step and collect snapshots.
    u_ts = torch.empty((B, S, S, steps+1), device=device, dtype=dtype)
    u_ts[:, :, :, 0] = u0

    # Broadcast shapes
    lam2d = lam2d.unsqueeze(0)                                 # (1,S,S)
    alpha = alpha.view(B, 1, 1).to(device=device, dtype=dtype) # (B,1,1)
    decay = torch.exp(-alpha * lam2d * dt)                     # (B,S,S)

    V_hat_t = V_hat.clone()
    for n in range(1, steps+1):
        V_hat_t = V_hat_t * decay                              # (B,S,S)
        v_t = sine2d_inverse(V_hat_t, S_dst)                       # (B,S,S)
        u_t = v_t + w                                          # add the lift back
        u_ts[:, :, :, n] = u_t

    return u_ts

# -------------------------
# Example usage
# -------------------------
@torch.no_grad()
def generate_heat(
        N: int,                 # Total number of samples
        B: int,                 # Batch size
        S: int,                 # grid size (S x S)
        steps: int,             # number of time steps to simulate
        dt: float,              # time step size
        Lx: float = 1.0,        # domain size in x
        Ly: float = 1.0,        # domain size in y
        alpha_logrange: tuple = (-2.0, 0.0),  # diffusivity range (log-uniform)
        device: str = "cpu",    # torch device
        dtype: torch.dtype = torch.float32,  # torch dtype
        ic_seed: int | None = None,         # random seed for IC generation (None = random each call
):
    
    X, Y = make_grid(S, Lx, Ly, device=device, dtype=dtype)     # (S,S)
    S_dst, lam2d = dirichlet_sine_basis(S, Lx, Ly, device=device, dtype=dtype)  # (S,S), (S,S)

    U = torch.empty((N, 1, S, S, steps+1), dtype=dtype)  
    A = torch.empty((N, 1, S, S), dtype=dtype)
    labels = torch.empty(N, dtype=dtype)

    for i in range(N // B + 1):
        if i == N // B:
            B = N % B
            if B == 0:
                break

        alpha = torch.exp(torch.empty(B).uniform_(*alpha_logrange)).to(device=device, dtype=dtype)      # log-uniform in [1e-2, 1]
        a = torch.empty(B).uniform_(-0.5, 0.5).to(device=device, dtype=dtype)
        b = torch.empty(B).uniform_(-0.5, 0.5).to(device=device, dtype=dtype)
        c = torch.empty(B).uniform_(-0.5, 0.5).to(device=device, dtype=dtype)

        u_ts = heat_timeseries_linear_bc(
            B=B, steps=steps, dt=dt, alpha=alpha, a=a, b=b, c=c,
            S=S, X=X, Y=Y, S_dst=S_dst, lam2d=lam2d,
            device=device, dtype=dtype, ic_seed=ic_seed,
        )

        A[i*B:(i+1)*B, 0, :, :] = u_ts[..., 0].cpu()
        U[i*B:(i+1)*B, 0, :, :, :] = u_ts.cpu()
        labels[i*B:(i+1)*B] = alpha.cpu()

        t_steps = np.arange(steps+1) * dt

    return U.numpy(), A.numpy(), t_steps, labels.numpy()

def main():
    B = 64
    N = 1000
    S = 64
    steps = 64
    T = 0.2
    dt = T / steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    seed = 123

    print(f"generating heat equation data.\n  N = {N}, S = {S}, B = {B}, T = {T:.4f}, steps = {steps}")

    U, A, t_steps, labels = generate_heat(
        N=N, B=B, S=S, steps=steps, dt=dt,
        device=device, dtype=dtype, ic_seed=seed,
    )
    print("computation done, saving data.")
    t_string = f"{T:.2f}".replace('.', '_')
    save_name = f"heat_eq_data_{N}_{S}_{S}_{steps}_{t_string}.npz"
    save_path = get_repo_root() / "data" / save_name
    np.savez(save_path, U=U, A=A, t_steps=t_steps, labels=labels)


if __name__ == "__main__":
    main()