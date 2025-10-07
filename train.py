from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import diffusion_pde as dpde
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):

    # load dataset configuration
    dataset_name = cfg.dataset.data.name
    datapath = cfg.dataset.data.datapath
    batch_size = cfg.dataset.training.batch_size
    test_split = cfg.dataset.training.test_split
    shuffle = cfg.dataset.training.shuffle
    generator_seed = cfg.dataset.training.generator_seed

    # load training configuration
    num_epochs = cfg.dataset.training.num_epochs
    learning_rate = cfg.dataset.training.learning_rate
    weight_decay = cfg.dataset.training.weight_decay

    # load model configuration
    chs = list(cfg.model.chs)
    noise_ch = cfg.model.noise_ch
    sigma_data = cfg.model.sigma_data

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb)

    # model save path:
    model_save_path = Path(cfg.model_save_path)

    # setup data loader
    generator = torch.Generator().manual_seed(generator_seed)

    trainloader, _ = dpde.datasets.get_dataloaders(
        datapath=datapath,
        batch_size=batch_size,
        test_split=test_split,
        shuffle=shuffle,
        generator=generator,
    )


    # set up for training
    ch_in = trainloader.dataset.dataset.input_ch
    label_ch = trainloader.dataset.dataset.label_ch
    chs = [ch_in] + chs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        wandb_kwargs=wandb_kwargs
    )

    torch.save(edm.state_dict(), model_save_path / f"edm_model_{dataset_name}.pth")

if __name__ == "__main__":
    main()






