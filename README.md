# dynamical-pde-diffusion
Specialkursus (Autumn 2025) - Predicting the dynamics of complex physical systems using denoising diffusion

# Conda environment from source with latest version of MagTense 2.2.5

```
conda create -n dyndiff python=3.13
conda activate dyndiff
python -m pip install hydra-core magtense omegaconf scipy torch wandb
python -m pip install -e .
```