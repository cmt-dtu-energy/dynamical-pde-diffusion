from pathlib import Path
import wandb
import hydra
import h5py
from omegaconf import DictConfig, OmegaConf
import diffusion_pde as dpde
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="test")
def main(cfg: DictConfig) -> None:
    # load dataset configuration
    dataset_name = cfg.dataset.data.name.lower()
    method = cfg.dataset.method.lower()

    # load model configuration
    model_name = cfg.model.name.lower()

    # load sampling configuration
    zeta_a = cfg.sampling_conf.zeta_a
    zeta_u = cfg.sampling_conf.zeta_u
    zeta_pde = cfg.sampling_conf.zeta_pde
    num_steps = cfg.sampling_conf.num_steps
    sigma_min = cfg.sampling_conf.sigma_min
    sigma_max = cfg.sampling_conf.sigma_max
    rho = cfg.sampling_conf.rho
    sample_shape = cfg.sampling_conf.sample_shape
    num_channels = cfg.sampling_conf.num_channels
    batch_size = cfg.sampling_conf.batch_size
    max_num_samples = cfg.sampling_conf.max_num_samples

    pretrained_path = Path(cfg.sampling_conf.pretrained_path)
    test_data_path = Path(cfg.sampling_conf.test_data_path)

    if "no_cond" in dataset_name or "no_time" in dataset_name:
        time_as_label = False
    else:
        time_as_label = True

    include_t0_as_target = cfg.dataset.data.get("include_t0_as_target", False)

    testloader = dpde.datasets.get_validation_dataloader(test_data_path, time_as_label=time_as_label, include_t0_as_target=include_t0_as_target)

    #load masks
    mask_a, mask_u = dpde.model_testing.get_masks_from_config(cfg)

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)


    # set up wandb run metadata
    job_type = "test"
    group = f"{model_name}"
    run_name = f"{dataset_name}/{method}/{model_name}/test".replace(" ", "-").replace("_", "-")
    tags = [dataset_name, model_name, job_type]
    #model_config = OmegaConf.to_container(cfg.model, resolve=True)

    config = OmegaConf.to_container(cfg, resolve=True)
    config["run_name"] = run_name

    wandb_kwargs.update({
        "name": run_name,
        "job_type": job_type,
        "group": group,
        "tags": tags,
        "config": config,
    })



    # load model and weights
    edm = dpde.utils.get_net_from_config(cfg)

    logger.info(f"Loading pretrained model from {pretrained_path}")
    edm.load_state_dict(torch.load(pretrained_path, weights_only=True))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load sampler:
    if "heat" in dataset_name:
        with h5py.File(dpde.utils.get_repo_root() / test_data_path, "r") as f:
            dx = f.attrs["dx"]
        pde_loss_fn = dpde.sampling.pde_losses.heat_loss2
        pde_loss_kwargs = {"dx": dx}
        out_and_grad_fn = dpde.sampling.X_and_dXdt_fd
    
    elif "llg" in dataset_name:
        pde_loss_fn = dpde.sampling.pde_losses.llg_loss2
        pde_loss_kwargs = {}
        out_and_grad_fn = dpde.sampling.X_and_dXdt_dummy
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    sampler = dpde.sampling.JointSampler(
        net=edm,
        device=device,
        sample_shape=sample_shape,
        num_channels=num_channels,
        num_samples=batch_size,
        ch_a=num_channels // 2,
        loss_fn=pde_loss_fn,
        loss_kwargs=pde_loss_kwargs,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        out_and_grad_fn=out_and_grad_fn,
    )

    # test model
    dpde.model_testing.test_loop(
        sampler=sampler,
        testloader=testloader,
        zeta_a=zeta_a,
        zeta_u=zeta_u,
        zeta_pde=zeta_pde,
        wandb_kwargs=wandb_kwargs,
        mask_a=mask_a,
        mask_u=mask_u,
        max_num_samples=max_num_samples,
    )


if __name__ == "__main__":
    main()