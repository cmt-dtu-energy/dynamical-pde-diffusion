import math
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import importlib
import torch
from diffusion_pde.models.pde_losses import PDE_LOSS_REGISTRY



def get_repo_root():
    try:
        import subprocess
        return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    except Exception:
        return Path(__file__).resolve().parent  # fallback


def get_loss_from_config(cfg):
    from diffusion_pde.models.loss import EDMLoss, EDMPhysicsLoss
    from diffusion_pde.models.pde_losses import heat_pde_loss

    if cfg.dataset.loss == "edm":
        loss_fn = EDMLoss()
    elif cfg.dataset.loss == "physics":
        raise NotImplementedError("Physics loss is not yet implemented in this function.")
        ch_a = cfg.dataset.net.in_ch // 2 if cfg.dataset.method == "joint" else 0
        loss_fn = EDMPhysicsLoss(
            pde_loss_fn=PDE_LOSS_REGISTRY[cfg.dataset.data.pde],
            pde_loss_kwargs={"dx": 1.0},
            pde_loss_coeff=0.01,
            ch_a=ch_a
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.dataset.loss}")
    return loss_fn


def get_net_from_config(cfg):
    from diffusion_pde.models import Unet, EDMUNet, EDMWrapper

    in_ch = cfg.dataset.net.in_ch
    label_ch = cfg.dataset.net.label_ch
    name = cfg.model.name.lower().replace(" ", "-").replace("_", "-")
    method = cfg.dataset.method
    if name == "unet-small":
        if method == "forward":
            raise ValueError("unet-small model is not compatible with 'forward' method since it requires obs channels.")
        chs = [in_ch] + list(cfg.model.chs)
        noise_ch = cfg.model.noise_ch
        unet = Unet(chs=chs, label_ch=label_ch, noise_ch=noise_ch)
    elif name == "unet-v2":
        if method == "forward":
            obs_ch = in_ch
        else:
            obs_ch = cfg.model.obs_ch
        unet = EDMUNet(
            img_channels=in_ch,
            label_dim=label_ch,
            obs_channels=obs_ch,
            base_channels=cfg.model.base_ch,
            channel_mults=cfg.model.ch_mults,
            num_res_blocks=cfg.model.n_res_blocks,
            dropout=cfg.model.dropout,
            sigma_emb_dim=cfg.model.noise_ch,
            emb_dim=cfg.model.emb_ch,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    edm = EDMWrapper(unet=unet, sigma_data=cfg.model.sigma_data)
    return edm


def get_function_from_path(path: str):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


class ResultsObject:
    """
    Utility class to manage paths for pretrained models and datasets based on configuration files.

    Parameters
    ----------
    cfg_path : str or Path
        Path to the configuration file used to set up the model and dataset.
        Can be either an absolute path or relative to the repository root.

    Attributes
    ----------
    root_path : Path
        Root directory of the repository.
    cfg : OmegaConf
        Loaded configuration object.
    save_name : str
        Constructed name for saving/loading models based on PDE, dataset, and model names.
    model_path : Path
        Path to the pretrained model file.
    data_path : Path
        Path to the dataset file.

    Methods
    -------
    get_wandb_model(root=None)
        Downloads the model from WandB artifact and returns the local path.
    """
    def __init__(self, cfg_path):
        cfg_path = Path(cfg_path).resolve()
        self.root_path = get_repo_root()
        if str(self.root_path) not in str(cfg_path):
            cfg_path = self.root_path / cfg_path
        self.cfg = OmegaConf.load(cfg_path)

        pde_name = self.cfg.dataset.data.pde.lower()
        data_name = self.cfg.dataset.data.name.lower()
        model_name = self.cfg.model.name.lower()
        self.save_name = f"{pde_name}_{data_name}_{model_name}".replace(" ", "_").replace("-", "_")

        self._model_path = self.root_path / "pretrained_models" / f"{self.save_name}.pth"
        self._data_path = self.root_path / self.cfg.dataset.data.datapath

        self._wandb_artifact_str = f"philiphohwy-danmarks-tekniske-universitet-dtu/dynamical-pde-diffusion/{self.save_name.replace('_', '-')}"

    @property
    def model_path(self):
        if self._model_path.exists():
            return self._model_path
        else:
            raise FileNotFoundError(f"Model path does not exist: {self._model_path}, consider loading from WandB artifact.")

    @property
    def data_path(self):
        if self._data_path.exists():
            return self._data_path
        else:
            raise FileNotFoundError(f"Data path does not exist: {self._data_path}.")

    def get_wandb_model(self, root="pretrained_models/wandb", version="latest"):
        api = wandb.Api()
        artifact = api.artifact(f"{self._wandb_artifact_str}:{version}")
        root = Path(root).resolve()
        if str(self.root_path) not in str(root):
            root = self.root_path / root

        artifact_dir = artifact.download(root=root)
        return Path(artifact_dir) / f"{self.save_name}.pth"
