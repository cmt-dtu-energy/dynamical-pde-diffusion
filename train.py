from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import diffusion_pde as dpde
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main(cfg):

    # load dataset configuration
    pde_name = cfg.dataset.data.pde.lower()
    dataset_name = cfg.dataset.data.name.lower()
    datapath = cfg.dataset.data.datapath
    batch_size = cfg.dataset.training.batch_size
    shuffle = cfg.dataset.training.shuffle

    # load training configuration
    num_epochs = cfg.dataset.training.num_epochs
    learning_rate = cfg.dataset.training.learning_rate
    weight_decay = cfg.dataset.training.weight_decay

    # load model configuration
    model_name = cfg.model.name.lower()
    chs = list(cfg.model.chs)
    noise_ch = cfg.model.noise_ch
    sigma_data = cfg.model.sigma_data

    # model save path:
    model_save_path = Path(cfg.model_save_path)

    trainloader = dpde.datasets.get_dataloader(
        datapath=datapath,
        batch_size=batch_size,
        shuffle=shuffle,
    )


    # set up for training
    in_ch = cfg.dataset.net.in_ch
    label_ch = cfg.dataset.net.label_ch
    chs = [in_ch] + chs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb)
    if wandb_kwargs["name"] == "None":
        wandb_kwargs["name"] = None
    job_type = "train"
    group = f"{pde_name}/{model_name}"
    run_name = f"{pde_name}/{dataset_name}/{model_name}".replace(" ", "-")
    tags = [pde_name, dataset_name, model_name]
    config = {
        "pde": pde_name,
        "dataset": dataset_name,
        "model": model_name,
        "chs": chs,
        "noise_ch": noise_ch,
        "label_ch": label_ch,
    }

    wandb_kwargs.update({
        "name": run_name,
        "job_type": job_type,
        "group": group,
        "tags": tags,
        "config": config,
    })

    save_name = f"{pde_name}_{dataset_name}_{model_name}.pth".replace(" ", "_")
    save_path = model_save_path / save_name

    unet = dpde.models.Unet(
        chs=chs,
        label_ch=label_ch,
        noise_ch=noise_ch,
    )

    edm = dpde.models.EDMWrapper(
        unet=unet,
        sigma_data=sigma_data
    )

    loss_fn = dpde.models.EDMLoss(sigma_data=sigma_data)

    edm.to(device)

    dpde.training.train(
        model=edm,
        dataloader=trainloader,
        loss_fn=loss_fn,
        device=device,
        epochs=num_epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        wandb_kwargs=wandb_kwargs,
        save_path=save_path,
    )
    #torch.save(edm.state_dict(), model_save_path / f"{pde_name}_{dataset_name}_{model_name}.pth")

if __name__ == "__main__":
    main()