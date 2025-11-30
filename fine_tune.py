import hydra
import h5py
import torch
import logging
import diffusion_pde as dpde
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main(cfg: DictConfig):

    try:
        pretrained_path = cfg.dataset.pretrained_path
        logger.info(f"loading pretrained model from: {pretrained_path}")
    except AttributeError:
        pretrained_path = None
        logger.info("no pretrained model path specified, exiting program")
        return

    # load dataset configuration
    dataset_name = cfg.dataset.data.name.lower()
    method = cfg.dataset.method.lower()

    # load training configuration
    num_epochs = cfg.dataset.training.num_epochs
    learning_rate = cfg.dataset.training.learning_rate
    weight_decay = cfg.dataset.training.weight_decay
    gradient_clipping = cfg.dataset.training.gradient_clipping
    val_interval = cfg.dataset.training.val_interval
    checkpoint_interval = cfg.dataset.training.checkpoint_interval
    ema_decay = cfg.dataset.training.ema_decay
    ema_warmup = cfg.dataset.training.ema_warmup
    ema_update_interval = cfg.dataset.training.ema_update_interval
    ema_device = cfg.dataset.training.ema_device

    # load model configuration
    model_name = cfg.model.name.lower()

    logger.info(f"training configuration for dataset: {dataset_name}, model: {model_name}, method: {method}")

    logger.info("loading training dataset and dataloader")
    trainloader, valloader = dpde.datasets.get_dataloaders(cfg)

    # set up for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    # load wandb configuration
    wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    if wandb_kwargs["name"] == "None":
        wandb_kwargs["name"] = None
    job_type = "train"
    group = f"{model_name}"
    run_name = f"{dataset_name}/{method}/{model_name}/fine-tune".replace(" ", "-").replace("_", "-")
    tags = [dataset_name, model_name, "fine-tune", method]
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

    if pretrained_path:
        edm.load_state_dict(torch.load(pretrained_path, weights_only=True))
        logger.info("pretrained model loaded successfully")

    loss_fn = dpde.utils.get_loss_from_config(cfg)

    edm.to(device)

    dpde.training.train(
        model=edm,
        dataloader=trainloader,
        valloader=valloader,
        loss_fn=loss_fn,
        device=device,
        epochs=num_epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        wandb_kwargs=wandb_kwargs,
        grad_clip=gradient_clipping,
        val_interval=val_interval,
        ema_decay=ema_decay,
        ema_warmup=ema_warmup,
        ema_update_interval=ema_update_interval,
        ema_device=ema_device,
        checkpoint_interval=checkpoint_interval,
    )

if __name__ == "__main__":
    main()