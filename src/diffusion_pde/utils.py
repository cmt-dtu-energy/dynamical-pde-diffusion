from pathlib import Path

def get_repo_root():
    try:
        import subprocess
        return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    except Exception:
        return Path(__file__).resolve().parent  # fallback
    

def get_net_from_config(cfg):
    from diffusion_pde.models import Unet, EDMWrapper

    in_ch = cfg.dataset.net.in_ch
    label_ch = cfg.dataset.net.label_ch
    chs = [in_ch] + list(cfg.model.chs)
    noise_ch = cfg.model.noise_ch
    if cfg.model.name.lower() == "unet small":
        unet = Unet(chs=chs, label_ch=label_ch, noise_ch=noise_ch)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    edm = EDMWrapper(unet=unet, sigma_data=cfg.model.sigma_data)
    return edm