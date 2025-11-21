import hydra
import torch
import logging
import diffusion_pde as dpde
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main(cfg: DictConfig):

    # load dataset configuration
    dataset_name = cfg.dataset.data.name.lower()
    method = cfg.dataset.method.lower()

    # load training configuration
    num_epochs = cfg.dataset.training.num_epochs
    learning_rate = cfg.dataset.training.learning_rate
    weight_decay = cfg.dataset.training.weight_decay

    # load model configuration
    model_name = cfg.model.name.lower()

    logger.info(f"training configuration for dataset: {dataset_name}, model: {model_name}, method: {method}")

    logger.info("loading training dataset and dataloader")
    trainloader = dpde.datasets.get_dataloader(cfg)

    # set up for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    if wandb_kwargs["name"] == "None":
        wandb_kwargs["name"] = None
    job_type = "train"
    group = f"{model_name}"
    run_name = f"{dataset_name}/{method}/{model_name}".replace(" ", "-").replace("_", "-")
    tags = [dataset_name, model_name]
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

    logger.info("initializing model, loss function, and starting training")

    edm = dpde.utils.get_net_from_config(cfg)

    loss_fn = dpde.utils.get_loss_from_config(cfg)

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
    )

if __name__ == "__main__":
    main()