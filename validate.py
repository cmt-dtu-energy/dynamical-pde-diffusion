from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import diffusion_pde as dpde
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):

    # load model configuration
    chs = list(cfg.model.chs)
    noise_ch = cfg.model.noise_ch
    sigma_data = cfg.model.sigma_data

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb)
    if wandb_kwargs["name"] == "None":
        wandb_kwargs["name"] = None

    # model save path:
    model_save_path = Path(cfg.model_save_path)

    # set up for training
    in_ch = cfg.dataset.net.in_ch
    label_ch = cfg.dataset.net.label_ch
    chs = [in_ch] + chs

    unet = dpde.models.Unet(chs=chs, label_ch=label_ch, noise_ch=noise_ch)
    edm = dpde.models.EDMWrapper(unet=unet, sigma_data=sigma_data)

    



if __name__ == "__main__":
    main()