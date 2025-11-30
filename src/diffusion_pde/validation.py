import math
import time
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import logging
from diffusion_pde.utils import get_function_from_path
from diffusion_pde.sampling import EDMHeatSampler
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

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


def load_mask_from_dir(pth):
    pth = Path(pth)
    if pth.suffix == ".mat":
        from scipy.io import loadmat
        masks = loadmat(pth)
    elif pth.suffix in [".npz"]:
        masks = np.load(pth)
    else:
        raise ValueError(f"unsupported file format: {pth.suffix}")
    return masks

def validate_model(
    model: torch.nn.Module,
    sampling_cfg: DictConfig,
    observation_cfg: DictConfig,
    wandb_kwargs: dict,
):
    """
    Validate a trained diffusion PDE model using generated data.

    Parameters:
    -----------
    validation_cfg : DictConfig DEPRECATED
        Configuration for data generation and validation parameters.
    sampling_cfg : DictConfig
        Configuration for the sampling process.
    observation_cfg : DictConfig
        Configuration for observation masks and parameters.
    wandb_kwargs : dict
        Configuration for WandB logging.

    Returns:
    --------
    None
    """

    sample_shape = sampling_cfg.sample_shape
    batch_size = sampling_cfg.batch_size
    ch_a = sampling_cfg.ch_a
    loss_func = get_function_from_path(sampling_cfg.loss_func)

    logger.info("Loading validation data from HDF5...")
    #Us, As, tsteps, labels, attrs = get_data_from_hdf5(Path(observation_cfg.dataset_path))
    #dx = attrs["dx"]
    #dy = attrs["dy"]

    s_shape = (batch_size, *sample_shape)


    # CHANGE TO LOADING DATA FROM DATABASE INSTEAD OF GENERATING ON THE FLY
    # ---------------------------------------------------------------------
    # Validation data generated here
    ###logger.info("Generating validation data...")
    ###Us, As, tsteps, labels = data_gen_wrapper(validation_cfg)   # tensors with dtype float64
    logger.info(f"Data min: {Us.min().item():<.4f}, max: {Us.max().item():<.4f}, num NaNs: {torch.isnan(Us).sum().item()}")
    if torch.any(torch.isinf(Us)):
        logger.error("Generated data contains infinite values! Exiting validation.")
        return
    elif torch.any(torch.isnan(Us)):
        logger.error("Generated data contains NaN values! Exiting validation.")
        return
    elif torch.any(Us > 1e3):
        logger.warning("Generated data contains very large values (>1e3). Check data generation process.")
    # ---------------------------------------------------------------------
    # ---------------------------- END CHANGES ----------------------------
    N = observation_cfg.num_validate
    logger.info(f"Validating on {N} samples out of {Us.shape[0]} available.")
    Us = Us[:N]
    As = As[:N]
    labels = labels[:N]
    
    num_tsteps = tsteps.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mask_a = None
    mask_u = None
    if observation_cfg.masks_path is not None:
        logger.info("Trying to load given masks...")
        try:
            masks = load_mask_from_dir(Path(observation_cfg.masks_path))
            mask_a = torch.tensor(masks["mask_a"])
            mask_u = torch.tensor(masks["mask_u"])
            logger.info("Successfully loaded masks")
        except Exception as e:
            logger.warning("Could not resolve mask path, defaulting to random masks.")
            pass

    if mask_a is None and mask_u is None:
        logger.info(f"Generating random masks for observations with fractions (interior, boundary): {observation_cfg.interior_a}, {observation_cfg.boundary_a} (a) and {observation_cfg.interior_u}, {observation_cfg.boundary_u} (u)")
        # set up observation masks
        interior_a = random_interior_mask(sample_shape[1], sample_shape[2], frac_obs=observation_cfg.interior_a)
        boundary_a = random_boundary_mask(sample_shape[1], sample_shape[2], frac_obs=observation_cfg.boundary_a)

        if observation_cfg.same_interior:
            interior_u = interior_a
        else:
            interior_u = random_interior_mask(sample_shape[1], sample_shape[2], frac_obs=observation_cfg.interior_u)

        if observation_cfg.same_boundary:
            boundary_u = boundary_a
        else:
            boundary_u = random_boundary_mask(sample_shape[1], sample_shape[2], frac_obs=observation_cfg.boundary_u)

        mask_a = combine_masks(interior_a, boundary_a)
        mask_u = combine_masks(interior_u, boundary_u)


    t_batch_all = tsteps.view(num_tsteps, 1, 1).expand(num_tsteps, batch_size, 1)
    mse = torch.zeros(N, num_tsteps, batch_size, sample_shape[0], device=device)    # shape: (N, time_steps, batch_size, channels)

    with wandb.init(**wandb_kwargs) as run:
        logger.info("Initialized WandB run.")
        run.config.update({
            #"validation_config": OmegaConf.to_container(validation_cfg, resolve=True),
            "sampling_config": OmegaConf.to_container(sampling_cfg, resolve=True),
            "observation_config": OmegaConf.to_container(observation_cfg, resolve=True)
        })
        t1_outer = time.perf_counter()
        for i in range(N):
            lbl = labels[i]
            #lbl = lbl.repeat(batch_size, 1)
            lbl = lbl.unsqueeze(0).expand(batch_size, -1)
            t1_inner = time.perf_counter()
            for j in range(num_tsteps):
                obs_a = As[i]
                obs_u = Us[i, ..., j]
                loss_fn_kwargs = {
                    "obs_a": obs_a,
                    "obs_u": obs_u,
                    "mask_a": mask_a,
                    "mask_u": mask_u,
                    "dx": dx,
                    "dy": dy,
                    "ch_a": ch_a,
                    "labels": lbl,
                }
                #t_lbl = tsteps[j].unsqueeze(0).repeat(batch_size, 1)
                t_lbl = t_batch_all[j]
                lbls = torch.cat([t_lbl, lbl], dim=1).to(device)

                samples, losses = edm_sampler(
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
                    return_losses=False,
                    debug=False,
                )
                obs = torch.cat([obs_a, obs_u], dim=0).to(torch.float32).to(device)
                mse[i, j] = torch.mean((samples - obs.unsqueeze(0)) ** 2, dim=(2, 3)).detach()   # mean squared error per channel - shape: (batch_size, channels)
                t2_inner = time.perf_counter()

            #if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"Completed validation of sample {i+1}/{N} in {t2_inner - t1_inner:.2f}s:\tMSE: {mse[i].mean().item()}")
            
        t2_outer = time.perf_counter()
        logger.info(f"Completed validation in {(t2_outer - t1_outer) / 60.:.2f} minutes.")

        for t in range(tsteps.shape[0]):
            run.log({"Error/validation/time-step/mse": mse[:, t].mean().item()}, step=t)     

        mse_batches = mse.mean(dim=2)         # mse over batches - shape (N, time_steps, channels)
        rmse_batches = mse_batches.clamp_min(0).sqrt()  # mean across batches - shape: (N, time_steps, channels)
        mu = rmse_batches.mean(axis=0).cpu().numpy()  # mean across samples - shape: (time_steps, channels)
        std = rmse_batches.std(dim=0).cpu().numpy()  # std across samples - shape: (time_steps, channels)

        if not torch.isfinite(mse_batches).all():
            logger.error("mse is not finite!")

        art_name = f"{run.config['pde']}-{run.config['dataset']}-{run.config['model']}-validation-errors".lower().replace(" ", "-").replace("_", "-")
        save_path = f"{art_name.replace('-', '_')}.npy"         
        np.save(save_path, mse.cpu().numpy())
        logger.info("Successfully stored errors.")

        artifact = wandb.Artifact(name=art_name, type="error", metadata=dict(run_id=run.id, **run.config))
        artifact.add_file(str(save_path))
        run.log_artifact(artifact)
        logger.info("Successfully logged errors to WandB.")

        t = tsteps.detach().cpu().numpy()
        for i in range(mu.shape[1]):
            fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
            ax.plot(t, mu[:, i], label="error", color="C0")
            ax.fill_between(t, mu[:, i] - std[:, i], mu[:, i] + std[:, i], color="C0", alpha=0.2, label="Std Dev")
            ax.scatter(t, mu[:, i], c="red", s=2, label="datapoints", zorder=10)
            ax.set(title="Validation RMSE over Time", xlabel="Time", ylabel="RMSE")
            ax.legend()
            plt.savefig(f"error_channel_{i}.png")
            run.log({f"Figures/Validate/Channel_{i}": wandb.Image(fig)})
        plt.close(fig)
        logger.info("Successfully generated figures.")

