from pathlib import Path
from diffusion_pde.datasets import DiffusionDataset

def get_repo_root():
    try:
        import subprocess
        return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    except Exception:
        return Path(__file__).resolve().parent  # fallback