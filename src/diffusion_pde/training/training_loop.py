import torch
import time
import wandb
from pathlib import Path
from diffusion_pde.models.loss import Loss
from diffusion_pde.utils import get_repo_root


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    wandb_kwargs: dict = {},
):  
    '''
    training function for diffusion model
    '''
    log_dir = get_repo_root() / "logs"
    wandb_kwargs.update({"dir": str(log_dir)})
    
    with wandb.init(**wandb_kwargs) as run:

    # log hyperparameters
        run.config.update({
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs
        })

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, labels) in enumerate(dataloader):
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                loss = loss_fn(model, data, labels).sum()
                loss.backward()

                running_loss += loss.item()
                optimizer.step()
            
            epoch_loss = running_loss / len(dataloader)

            run.log({"Loss/train/epoch": epoch_loss})

