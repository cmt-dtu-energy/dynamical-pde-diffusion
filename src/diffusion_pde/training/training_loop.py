import torch
import wandb
import logging
from pathlib import Path
from diffusion_pde.models.loss import Loss
from diffusion_pde.models.nets import EMAWrapper

logger = logging.getLogger(__name__)

def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    wandb_kwargs: dict,
    save_path: Path | None = None,
    grad_clip: float | None = None,
    val_interval: int | None = None,
    ema_decay: float | None = None,
    ema_warmup: int = 0,
    ema_update_interval: int = 1,
    ema_device: torch.device | str = "cpu",
    checkpoint_interval: int | None = None,
):  
    '''
    training function for diffusion model
    '''
    if save_path is None:
        save_path = Path().cwd() / "model.pth"
    if checkpoint_interval is not None:
        checkpoint_dir = save_path.parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)


    logger.info("initializing wandb run for training")
    with wandb.init(**wandb_kwargs) as run:
        global_step = -1
        best_val_loss = float("inf")

    # log hyperparameters
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        ema = None
        if ema_decay is not None:
            logger.info(f"initializing EMA model with decay {ema_decay:.4f} and warmup: {ema_warmup} steps")
            ema = EMAWrapper(
                model, 
                ema_decay=ema_decay, 
                ema_device=ema_device,
                update_every=ema_update_interval, 
                warmup_steps=ema_warmup
                )
        
        logger.info("starting training loop")
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            for i, kwargs in enumerate(dataloader):
                global_step += 1
                for k in kwargs:
                    kwargs[k] = kwargs[k].to(device)
                
                X = kwargs.pop("X")
                labels = kwargs.pop("labels")

                optimizer.zero_grad()
                loss = loss_fn(model, X, labels, run=run, global_step=global_step, **kwargs).mean()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                running_loss += loss.item()

                if ema is not None:
                    ema.update()
            
            epoch_loss = running_loss / len(dataloader)

            run.log({"Loss/train/epoch": epoch_loss, "epoch": epoch}, step=global_step)

            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")


            if val_interval is not None and (epoch + 1) % val_interval == 0:
                logger.info("starting validation loop")
                val_model = model if ema is None else ema.ema_model.to(device)
 
                val_model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for j, val_kwargs in enumerate(valloader):
                        for k in val_kwargs:
                            val_kwargs[k] = val_kwargs[k].to(device)
                        
                        X_val = val_kwargs.pop("X")
                        labels_val = val_kwargs.pop("labels")

                        loss_val = loss_fn(val_model, X_val, labels_val, **val_kwargs).mean()
                        val_running_loss += loss_val.item()
                
                val_loss = val_running_loss / len(valloader)
                run.log({"Loss/val": val_loss, "epoch": epoch}, step=global_step)
                logger.info(f"  Validation Loss: {val_loss:.6f}")

                if ema is not None:
                    ema.ema_model.to(ema_device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"  New best model found at epoch {epoch+1}")
                    if ema is not None:
                        logger.info("  saving EMA model as best model")
                        ema_save_path = checkpoint_dir / f"ema_model_best.pth"
                        torch.save(ema.ema_model.state_dict(), ema_save_path)

            if checkpoint_interval is not None and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"model_epoch_{epoch + 1}.pth"
                logger.info(f"saving checkpoint at epoch {epoch+1} to {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)

        logger.info("training complete, saving model")
        torch.save(model.state_dict(), save_path)
        if ema is not None:
            ema_save_path = save_path.parent / f"ema_{save_path.name}"
            torch.save(ema.ema_model.state_dict(), ema_save_path)

        logger.info("logging model artifact to wandb")
        art_name = run.config["run_name"].replace("/", "-")
        logger.info("artifact name: %s", art_name)
        artifact = wandb.Artifact(name=art_name, type="model", metadata=dict(run_id=run.id, **run.config))
        artifact.add_file(str(save_path))
        if ema is not None:
            artifact.add_file(str(ema_save_path))
        run.log_artifact(artifact)
        logger.info("model artifact logged successfully, training finished")