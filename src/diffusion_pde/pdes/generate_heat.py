import math
import torch
import numpy as np
from diffusion_pde.utils import get_repo_root
from diffusion_pde.pdes import save_dataset, save_data

# -------------------------
# Utilities: grid & sine basis
# -------------------------
def make_grid(S: int = 64, Lx: float = 1.0, Ly: float = 1.0, *, device=None, dtype=None):
    """
    Returns full grid X,Y of shape (S, S), including boundary rows/cols.
    """
    x = torch.linspace(0., Lx, S, device=device, dtype=dtype)
    y = torch.linspace(0., Ly, S, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')  # (S,S)
    return X, Y

def dirichlet_sine_basis(N_int: int, Lx: float = 1.0, Ly: float = 1.0, *, device=None, dtype=None):
    """
    Orthonormal DST-II-like matrix for N_int INTERIOR points per axis (Dirichlet).
    Also returns 2D Laplacian eigenvalues lam2d for those interior modes.

    S[j,n] = sqrt(2/(N_int+1)) * sin(pi * j * n / (N_int+1)), j,n=1..N_int
    """
    if N_int <= 0:
        raise ValueError(f"N_int must be >= 1, got {N_int}")
    n = torch.arange(1, N_int + 1, device=device, dtype=dtype).view(1, -1)  # (1,N_int)
    j = torch.arange(1, N_int + 1, device=device, dtype=dtype).view(-1, 1)  # (N_int,1)

    S = torch.sqrt(torch.tensor(2.0 / (N_int + 1), device=device, dtype=dtype)) * \
        torch.sin(math.pi * j * n / (N_int + 1))  # (N_int, N_int)

    # eigenvalues for 1D dirichlet modes in [0, L]
    kx = math.pi * n.squeeze(0) / Lx  # (N_int,)
    ky = math.pi * n.squeeze(0) / Ly  # (N_int,)
    lam_x = kx ** 2
    lam_y = ky ** 2
    lam2d = lam_y.view(N_int, 1) + lam_x.view(1, N_int)  # (N_int, N_int)
    return S, lam2d

def sine2d_forward(U_interior: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    2D forward DST on INTERIOR slice: U_hat = S @ U @ S^T
    U_interior: (..., N_int, N_int), S: (N_int, N_int)
    """
    tmp = torch.matmul(S, U_interior)       # (..., N_int, N_int)
    U_hat = torch.matmul(tmp, S.t())
    return U_hat

def sine2d_inverse(U_hat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    2D inverse DST on INTERIOR slice: U = S^T @ U_hat @ S
    """
    tmp = torch.matmul(U_hat, S)            # (..., N_int, N_int)
    U = torch.matmul(S.t(), tmp)
    return U

# -------------------------
# Linear BC lift: w(x,y) = a + b x + c y
# -------------------------
def linear_bc_field(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, X: torch.Tensor, Y: torch.Tensor):
    """
    a,b,c: (B,)  |  X,Y: (S,S)  -> returns w: (B,S,S)
    """
    return a.view(-1, 1, 1) + b.view(-1, 1, 1) * X + c.view(-1, 1, 1) * Y

# -------------------------
# Random Gaussian blob IC (batched)
# -------------------------
def random_gaussian_blobs(
    B: int, S: int, X: torch.Tensor, Y: torch.Tensor, *,
    n_blobs=(1, 3), amp_range=(0.5, 1.0), sigma_range=(0.03, 0.15),
    device=None, dtype=None, seed=None
):
    """
    Returns (B,S,S) blobs on the provided grid (assumed in [0,1]x[0,1]).
    """
    if seed is not None:
        torch.manual_seed(seed)
    X = X.to(device=device, dtype=dtype)
    Y = Y.to(device=device, dtype=dtype)

    out = torch.zeros((B, S, S), device=device, dtype=dtype)
    BLOB_MIN, BLOB_MAX = map(int, n_blobs)

    for b in range(B):
        k = torch.randint(BLOB_MIN, BLOB_MAX + 1, (1,), device=device).item()
        u = torch.zeros((S, S), device=device, dtype=dtype)
        for _ in range(k):
            cx = torch.rand((), device=device, dtype=dtype)
            cy = torch.rand((), device=device, dtype=dtype)
            sx = torch.empty((), device=device, dtype=dtype).uniform_(*sigma_range)
            sy = torch.empty((), device=device, dtype=dtype).uniform_(*sigma_range)
            amp = torch.empty((), device=device, dtype=dtype).uniform_(*amp_range)
            ga = amp * torch.exp(-((X - cx) ** 2 / (2 * sx ** 2) + (Y - cy) ** 2 / (2 * sy ** 2)))
            if torch.rand((), device=device) < 0.5:
                ga = -ga
            u = u + ga
        out[b] = u
    return out

# -------------------------
# Heat evolution with linear Dirichlet BCs via lifting + interior DST
# -------------------------
@torch.no_grad()
def heat_timeseries_linear_bc(
    B: int,
    steps: int,
    dt: torch.Tensor,         # (steps,)
    alpha: torch.Tensor,      # (B,)
    a: torch.Tensor,          # (B,)
    b: torch.Tensor,          # (B,)
    c: torch.Tensor,          # (B,)
    S: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    S_int: torch.Tensor,      # (S-2, S-2) interior DST matrix
    lam2d_int: torch.Tensor,  # (S-2, S-2) interior eigenvalues
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    ic_seed: int | None = None,
):
    """
    Returns: u_ts (B, S, S, steps+1) satisfying u|∂Ω = a + b x + c y.
    Uses interior-size DST (S-2) for correct Dirichlet evolution.
    """
    device = torch.device(device)
    dt = dt.to(device=device, dtype=dtype)

    # Lift for BCs
    w = linear_bc_field(a, b, c, X, Y)  # (B,S,S)

    # Initial condition (random blobs), then force boundary to w
    u0_raw = random_gaussian_blobs(B, S, X, Y, device=device, dtype=dtype, seed=ic_seed)
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
    u_ts = torch.empty((B, S, S, steps + 1), device=device, dtype=dtype)
    u_ts[..., 0] = u0

    # Broadcastables
    alpha = alpha.view(B, 1, 1).to(device=device, dtype=dtype)  # (B,1,1)
    lam2d = lam2d_int.to(device=device, dtype=dtype)            # (S-2,S-2)

    V_hat_t = V_hat.clone()
    for n in range(1, steps + 1):
        decay = torch.exp(-alpha * lam2d * dt[n - 1])    # (B,S-2,S-2) via broadcast
        V_hat_t = V_hat_t * decay
        v_in = sine2d_inverse(V_hat_t, S_int)            # (B,S-2,S-2)

        # re-embed interior & add lift
        v_full = torch.zeros((B, S, S), device=device, dtype=dtype)
        v_full[:, 1:-1, 1:-1] = v_in
        u_t = v_full + w
        u_ts[..., n] = u_t

    return u_ts

# -------------------------
# Public generator
# -------------------------
@torch.no_grad()
def generate_heat(
    N: int,                  # total samples
    B: int,                  # batch size (can be >= N)
    S: int,                  # grid size (SxS)
    steps: int,              # number of time steps
    dt: torch.Tensor,        # (steps,)
    Lx: float = 1.0,
    Ly: float = 1.0,
    alpha_logrange: tuple = (-2.0, 0.0),  # log-uniform range for alpha in [1e-2, 1]
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    ic_seed: int | None = None,
):
    # full grid (includes boundaries)
    X, Y = make_grid(S, Lx, Ly, device=device, dtype=dtype)

    # interior DST (S-2)
    N_int = S - 2
    if N_int <= 0:
        raise ValueError("S must be >= 3 for an interior grid.")
    S_int, lam2d_int = dirichlet_sine_basis(N_int, Lx, Ly, device=device, dtype=dtype)

    # allocate outputs on CPU (like your previous return types)
    U = torch.empty((N, 1, S, S, steps + 1), dtype=dtype)
    A = torch.empty((N, 1, S, S), dtype=dtype)
    labels = torch.empty(N, dtype=dtype)

    # iterate in chunks; handles any B vs N (including B >= N)
    start = 0
    while start < N:
        this_B = min(B, N - start)

        # per-chunk parameters on device
        alpha = torch.exp(torch.empty(this_B, device=device, dtype=dtype).uniform_(*alpha_logrange))
        a = torch.empty(this_B, device=device, dtype=dtype).uniform_(-0.5, 0.5)
        b = torch.empty(this_B, device=device, dtype=dtype).uniform_(-0.5, 0.5)
        c = torch.empty(this_B, device=device, dtype=dtype).uniform_(-0.5, 0.5)

        # integrate on device
        u_ts = heat_timeseries_linear_bc(
            B=this_B, steps=steps, dt=dt,
            alpha=alpha, a=a, b=b, c=c,
            S=S, X=X, Y=Y, S_int=S_int, lam2d_int=lam2d_int,
            device=device, dtype=dtype, ic_seed=ic_seed,
        )  # (this_B, S, S, steps+1)

        end = start + this_B
        A[start:end, 0] = u_ts[..., 0].cpu()
        U[start:end, 0] = u_ts.cpu()
        labels[start:end] = alpha.cpu()
        start = end

    # build time stamps: length steps+1, starting at 0
    t_steps = np.concatenate((np.zeros(1, dtype=np.float32), dt.detach().cpu().numpy().astype(np.float32).cumsum()))
    labels = labels.view(-1, 1)  # (N,1)

    return U.numpy(), A.numpy(), t_steps, labels.numpy()

# -------------------------
# Example usage
# -------------------------
def main():
    B = 50
    N = 5000
    S = 64
    steps = 64
    T = 0.5
    Lx, Ly = 1.0, 1.0
    alpha_logrange = (-2.5, 0.5)

    # log-scale time steps in (0, T]
    t_spacing = "linear"


    if t_spacing == "linear":
        TT = torch.linspace(0, T, steps + 1)   # for linear time
    elif t_spacing == "log":
        TT = torch.logspace(-4, math.log10(T), steps + 1)   # for logspaced time
    else: 
        raise ValueError(f"Unknown t_spacing: {t_spacing}")
    
    dt = TT[1:] - TT[:-1]                               # (steps,)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    seed = 12

    print(f"generating heat equation data (log time): N={N}, S={S}, B={B}, T={T:.4f}, steps={steps}")

    U, A, t_steps, labels = generate_heat(
        N=N, B=B, S=S, steps=steps, dt=dt,
        Lx=Lx, Ly=Ly, alpha_logrange=alpha_logrange,
        device=device, dtype=dtype, ic_seed=seed,
    )

    print("computation done, saving data.")

    save_name = f"heat_{t_spacing}t.hdf5"
    save_path = get_repo_root() / "data" / save_name

    save_data(
        filepath=str(save_path),
        A=A,
        U=U,
        labels=labels,
        t_steps=t_steps,
        T=T,
        dx=Lx / (S - 1),
        dy=Ly / (S - 1),
        name=save_name[:-5],
        description=f"2D heat equation with linear Dirichlet BCs, pseudospectral interior DST with lifting. {t_spacing}-spaced time.",
        S=S,
        Lx=Lx,
        Ly=Ly,
        alpha_logrange=alpha_logrange,
        steps=steps,
    )

if __name__ == "__main__":
    main()
