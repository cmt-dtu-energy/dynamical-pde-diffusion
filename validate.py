from pathlib import Path
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import diffusion_pde as dpde
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    # load dataset configuration
    pde_name = cfg.dataset.data.pde.lower()
    dataset_name = cfg.dataset.data.name.lower()

    # load model configuration
    model_name = cfg.model.name.lower()

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb)
    if wandb_kwargs["name"] == "None":
        wandb_kwargs["name"] = None

    job_type = "validate"
    group = f"{pde_name}/{model_name}"
    run_name = f"{pde_name}/{dataset_name}/{model_name}/validate".replace(" ", "-")
    tags = [pde_name, dataset_name, model_name]
    config = {
        "pde": pde_name,
        "dataset": dataset_name,
        "model": model_name,
    }

    wandb_kwargs.update({
        "name": run_name,
        "job_type": job_type,
        "group": group,
        "tags": tags,
        "config": config,
    })


    # Validation data generated here
    Us, As, tsteps, labels = dpde.validation.data_gen_wrapper(cfg.dataset.validation)
    N = Us.shape[0]

    device = torch.device(cfg.dataset.validation.device)
    edm = dpde.utils.get_net_from_config(cfg).to(device)

    with wandb.init(**wandb_kwargs) as run:
        pass

    

if __name__ == "__main__":
    main()