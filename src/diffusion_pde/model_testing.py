import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import logging
from diffusion_pde.sampling import Sampler, sampling_context
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

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


def get_masks_from_config(cfg):

    sample_shape = cfg.sampling_conf.sample_shape
    interior_a = cfg.observations.interior_a
    boundary_a = cfg.observations.boundary_a
    interior_u = cfg.observations.interior_u
    boundary_u = cfg.observations.boundary_u
    same_interior = cfg.observations.same_interior
    same_boundary = cfg.observations.same_boundary

    mask_a = None
    mask_u = None

    if mask_a is None and mask_u is None:
        logger.info(f"Generating random masks for observations with fractions (interior, boundary): {interior_a}, {boundary_a} (a) and {interior_u}, {boundary_u} (u)")
        # set up observation masks
        interior_a = random_interior_mask(sample_shape[0], sample_shape[1], frac_obs=interior_a)
        boundary_a = random_boundary_mask(sample_shape[0], sample_shape[1], frac_obs=boundary_a)

        if same_interior:
            interior_u = interior_a
        else:
            interior_u = random_interior_mask(sample_shape[0], sample_shape[1], frac_obs=interior_u)

        if same_boundary:
            boundary_u = boundary_a
        else:
            boundary_u = random_boundary_mask(sample_shape[0], sample_shape[1], frac_obs=boundary_u)

        mask_a = combine_masks(interior_a, boundary_a)
        mask_u = combine_masks(interior_u, boundary_u)

    return mask_a, mask_u


def test_loop(
    sampler: Sampler,
    testloader: torch.utils.data.DataLoader,
    zeta_a: float,
    zeta_u: float,
    zeta_pde: float,
    wandb_kwargs: dict,
    mask_a: torch.Tensor | None = None,
    mask_u: torch.Tensor | None = None,
    num_steps: int = 50,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    max_num_samples: int = 1000,
):

    if mask_a is None:
        mask_a = torch.zeros(sampler.num_channels // 2, sampler.sample_shape[0], sampler.sample_shape[1], dtype=torch.bool)
    if mask_u is None:
        mask_u = torch.zeros(sampler.num_channels // 2, sampler.sample_shape[0], sampler.sample_shape[1], dtype=torch.bool)

    # preallocate error tensor:
    num_samples = min(len(testloader.dataset), max_num_samples)
    MAE = torch.empty((num_samples, sampler.num_channels, *sampler.sample_shape))
    denom_abs = torch.empty((num_samples, sampler.num_channels, *sampler.sample_shape))
    denom_range = torch.empty((num_samples, sampler.num_channels))
    std = torch.empty((num_samples, sampler.num_channels, *sampler.sample_shape))
    with wandb.init(**wandb_kwargs) as run:
        logger.info("Initialized WandB run.")
        with sampling_context(sampler):
            for i, batch in tqdm(enumerate(testloader), total=num_samples, desc="Testing"):
                if i >= max_num_samples:
                    break
                
                A = batch["A"]  # (1, C, H, W)
                U = batch["U"]  # (1, C, H, W)
                labels = batch["labels"]  # (1, label_dim) or None
                if labels is not None:
                    labels = labels.expand(sampler.num_samples, -1)  # (num_samples, label_dim)

                samples, _ = sampler.sample_joint(
                    labels=labels,
                    obs_a=A,
                    obs_u=U,
                    mask_a=mask_a,
                    mask_u=mask_u,
                    zeta_a=zeta_a,
                    zeta_u=zeta_u,
                    zeta_pde=zeta_pde,
                    num_steps=num_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=rho,
                    return_losses=False,
                ) # (B, 2C, H, W)
                obs = torch.cat([A, U], dim=1)  # (1, 2C, H, W)

                mae = (obs - samples).abs().mean(dim=0)  # mean absolute error over batch dimension
                d_abs = obs.abs()# mean absolute value of observations over batch dimension
                d_range = obs.squeeze(0).amax(dim=(-2, -1)) - obs.squeeze(0).amin(dim=(-2, -1))  # range of observations over spatial dimensions per channel
                sample_std = samples.std(dim=0)  # std dev of samples over batch dimension
                MAE[i * batch["A"].shape[0] : i * batch["A"].shape[0] + mae.shape[0]] = mae
                denom_abs[i * batch["A"].shape[0] : i * batch["A"].shape[0] + mae.shape[0]] = d_abs # denominator for relative error
                denom_range[i * batch["A"].shape[0] : i * batch["A"].shape[0] + mae.shape[0]] = d_range  # denominator for normalized error
                std[i * batch["A"].shape[0] : i * batch["A"].shape[0] + mae.shape[0]] = sample_std  # std dev of samples

                run.log({
                    "rel MAE": (mae / d_range.unsqueeze(-1).unsqueeze(-1)).mean().item(),
                    "sample rel std": (sample_std / d_range.unsqueeze(-1).unsqueeze(-1)).mean().item()
                })


        if not torch.isfinite(MAE).all():
            logger.error("MAE is not finite!")


        save_path = "validation_data.npz"         
        np.savez(save_path, MAE=MAE.numpy(), denom_abs=denom_abs.numpy(), denom_range=denom_range.numpy(), std=std.numpy())
        logger.info("Successfully stored errors.")

        rel_error = MAE / denom_range.unsqueeze(-1).unsqueeze(-1)
        ch_rel_error = rel_error.mean(dim=(0, 2, 3))
        for i in range(sampler.num_channels):
            logger.info(f"  Channel {i}: Mean Relative Error: {ch_rel_error[i]:.4f}")

        logger.info("Test loop completed.")
