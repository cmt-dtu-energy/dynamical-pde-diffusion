from pathlib import Path
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import diffusion_pde as dpde
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="validate")
def main(cfg):
    # load dataset configuration
    pde_name = cfg.dataset.data.pde.lower()
    dataset_name = cfg.dataset.data.name.lower()

    # load model configuration
    model_name = cfg.model.name.lower()

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)

    job_type = "validate"
    group = f"{pde_name}/{model_name}"
    run_name = f"{pde_name}/{dataset_name}/{model_name}/validate".replace(" ", "-")
    tags = [pde_name, dataset_name, model_name, "validate"]
    
    config = OmegaConf.to_container(cfg, resolve=True)
    config["run_name"] = run_name

    wandb_kwargs.update({
        "name": run_name,
        "job_type": job_type,
        "group": group,
        "tags": tags,
        "config": config,
    })

    edm = dpde.utils.get_net_from_config(cfg)

    model_save_path = Path(cfg.pretrained_path)
    logger.info(f"Loading pretrained model from {model_save_path}")

    edm.load_state_dict(torch.load(model_save_path, weights_only=True))

    dpde.validation.validate_model(
        model=edm,
        #validation_cfg=cfg.dataset.validation,
        sampling_cfg=cfg.dataset.sampling,
        observation_cfg=cfg.observations,
        wandb_kwargs=wandb_kwargs,
    )


if __name__ == "__main__":
    main()