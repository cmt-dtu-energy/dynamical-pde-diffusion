import math
import torch
import wandb
from diffusion_pde.utils import get_function_from_path
from diffusion_pde.sampling import edm_sampler
from omegaconf import DictConfig, OmegaConf


def data_gen_wrapper(validation_cfg: DictConfig):
    """
    wrapper for PDE data generating functions for model validation

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing data generation function path and arguments.

    Returns
    -------
    """
    data_gen_func = get_function_from_path(validation_cfg.data_gen_func)
    func_args = validation_cfg.func_kwargs

    N = func_args.N,
    B = func_args.B,
    Lx = func_args.Lx,
    Ly = func_args.Ly,
    Nx = func_args.Nx,
    Ny = func_args.Ny,
    steps = func_args.steps,
    T = func_args.T,
    device = func_args.device,
    ic_seed = func_args.seed

    if "generate_heat" in validation_cfg.data_gen_func:
        time_spacing = func_args.get("time_spacing", "linear")

        if time_spacing == "linear":
            time_steps = torch.linspace(0, T, steps + 1)
        elif time_spacing == "log":
            time_steps = torch.logspace(-4, math.log10(T), steps + 1)
        dt = time_steps[1:] - time_steps[:-1]
        Us, As, tsteps, labels = data_gen_func(N=N, B=B, S=Nx, steps=steps, dt=dt, Lx=Lx, Ly=Ly, device=device, ic_seed=ic_seed)

    elif "generate_llg" in validation_cfg.data_gen_func:
        raise NotImplementedError("LLG data generation not implemented in this wrapper yet.")
    
    else:
        raise ValueError(f"Unknown data generation function: {validation_cfg.data_gen_func}")

    return Us, As, tsteps, labels


def random_boundary_mask(H, W, *, frac_obs=0.5, n=None, device=None, generator=None, include_corners=True):
    """
    Generate a random binary mask for boundary observations.

    Parameters
    ----------
    H : int
        Height of the 2D grid.
    W : int
        Width of the 2D grid.
    frac_obs : float, optional
        Fraction of boundary points to observe (between 0 and 1), by default 0.5.
    n : int, optional
        Exact number of boundary points to observe. If provided, overrides frac_obs.
    device : torch.device, optional
        Device on which to create the mask tensor. If None, uses CPU.
    generator : torch.Generator, optional
        Random number generator for reproducibility.
    include_corners : bool, optional
        Whether to include corner points as possible observations, by default True.

    Returns
    -------
    torch.Tensor
        A binary mask tensor of shape (H, W) with True for observed boundary points and False elsewhere.
    """
    m = torch.zeros(H, W, dtype=torch.bool, device=device)
    m[[0, -1], :] = True
    m[:, [0, -1]] = True
    if not include_corners:
        m[0, 0] = m[0, -1] = m[-1, 0] = m[-1, -1] = False
    if n is None:
        n = int(frac_obs * (2 * H + 2 * W - 4))  # number of boundary pixels
    elif frac_obs == 1.0:
        return m
    elif frac_obs == 0.0:
        return torch.zeros(H, W, dtype=torch.bool, device=device)

    b = torch.where(m.flatten())[0]  # 1D indices of boundary pixels
    if n > b.numel():
        raise ValueError(f"n={n} > boundary points={b.numel()}")
    keep = b[torch.randperm(b.numel(), device=device, generator=generator)[:n]]

    m.zero_()
    m.view(-1)[keep] = True
    return m  # shape (H, W)


def random_interior_mask(H, W, *, frac_obs=0.5, n=None, device=None, generator=None):
    """
    Generate a random binary mask for interior observations.

    Parameters
    ----------
    H : int
        Height of the 2D grid.
    W : int
        Width of the 2D grid.
    frac_obs : float, optional
        Fraction of interior points to observe (between 0 and 1), by default 0.5.
    n : int, optional
        Exact number of interior points to observe. If provided, overrides frac_obs.
    device : torch.device, optional
        Device on which to create the mask tensor. If None, uses CPU.
    generator : torch.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    torch.Tensor
        A binary mask tensor of shape (H, W) with True for observed interior points and False elsewhere.
    """
    m = torch.zeros(H, W, dtype=torch.bool, device=device)
    m[1:-1, 1:-1] = True  # interior points

    if n is None:
        n = int(frac_obs * (H - 2) * (W - 2))  # number of interior pixels
    elif frac_obs == 1.0:
        return m
    elif frac_obs == 0.0:
        return torch.zeros(H, W, dtype=torch.bool, device=device)

    b = torch.where(m.flatten())[0]  # 1D indices of interior pixels
    if n > b.numel():
        raise ValueError(f"n={n} > interior points={b.numel()}")
    keep = b[torch.randperm(b.numel(), device=device, generator=generator)[:n]]

    m.zero_()
    m.view(-1)[keep] = True
    return m  # shape (H, W)


def combine_masks(*masks):
    """
    Combine multiple binary masks into a single mask using logical OR.

    Parameters
    ----------
    *masks : torch.Tensor
        Multiple binary mask tensors of the same shape.

    Returns
    -------
    torch.Tensor
        A combined binary mask tensor with True where any input mask is True.
    """
    if not masks:
        raise ValueError("At least one mask must be provided.")
    
    combined_mask = masks[0].clone()
    for m in masks[1:]:
        combined_mask |= m
    return combined_mask


def validate_model(
    model: torch.nn.Module,
    validation_cfg: DictConfig,
    sampling_cfg: DictConfig,
    wandb_kwargs: dict,
):
    """
    Validate a trained diffusion PDE model using generated data.
    """

    sample_shape = sampling_cfg.sample_shape
    batch_size = sampling_cfg.batch_size
    ch_a = sampling_cfg.ch_a
    loss_func = get_function_from_path(sampling_cfg.loss_func)

    dx = validation_cfg.func_kwargs.Lx / (sample_shape[1] - 1)
    dy = validation_cfg.func_kwargs.Ly / (sample_shape[2] - 1)

    s_shape = (batch_size, *sample_shape)

    # Validation data generated here
    Us, As, tsteps, labels = data_gen_wrapper(validation_cfg)
    N = Us.shape[0]

    device = torch.device(validation_cfg.device)
    model.to(device)

    # set up observation masks
    interior_a = random_interior_mask(sample_shape[1], sample_shape[2], frac_obs=validation_cfg.observations.interior_a)
    boundary_a = random_boundary_mask(sample_shape[1], sample_shape[2], frac_obs=validation_cfg.observations.boundary_a)

    if validation_cfg.observations.same_interior:
        interior_u = interior_a
    else:
        interior_u = random_interior_mask(sample_shape[1], sample_shape[2], frac_obs=validation_cfg.observations.interior_u)

    if validation_cfg.observations.same_boundary:
        boundary_u = boundary_a
    else:
        boundary_u = random_boundary_mask(sample_shape[1], sample_shape[2], frac_obs=validation_cfg.observations.boundary_u)

    mask_a = combine_masks(interior_a, boundary_a)
    mask_u = combine_masks(interior_u, boundary_u)


    with wandb.init(**wandb_kwargs) as run:
        run.config.update({
            "validation_config": OmegaConf.to_container(validation_cfg, resolve=True),
            "sampling_config": OmegaConf.to_container(sampling_cfg, resolve=True),
        })

        model.eval()
        errors = torch.zeros(N, tsteps.shape[0], batch_size, sample_shape[0], device=device)

        for i in range(N):
            lbl = labels[i]
            lbls = lbl.repeat(batch_size, 1)
            for j in range(tsteps.shape[0]):
                obs_a = As[i]
                obs_u = Us[i, ..., j]
                obs = torch.cat([obs_a, obs_u], dim=0).to(device)
                loss_fn_kwargs = {
                    "obs_a": obs_a,
                    "obs_u": obs_u,
                    "mask_a": mask_a,
                    "mask_u": mask_u,
                    "dx": dx,
                    "dy": dy,
                    "ch_a": ch_a,
                    "label": lbl,
                }
                t_labels = torch.full((sample_shape[0], 1), tsteps[j])
                lbls = torch.cat([t_labels, lbls], dim=1).to(device)

                samples, _ = edm_sampler(
                    net=model,
                    device=device,
                    sample_shape=s_shape,
                    loss_fn=loss_func,
                    loss_fn_kwargs=loss_fn_kwargs,
                    labels=lbls,
                    zeta_a=sampling_cfg.zeta_a,
                    zeta_u=sampling_cfg.zeta_u,
                    zeta_pde=sampling_cfg.zeta_pde,
                    num_steps=sampling_cfg.num_steps,
                    to_cpu=False,
                )

                errors[i, j, :] = torch.mean((samples - obs.unsqueeze(0)) ** 2, dim=(2, 3)).detach()

        
