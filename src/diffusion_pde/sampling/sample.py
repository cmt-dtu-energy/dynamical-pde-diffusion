import torch
import logging
import numpy as np
from torch.func import jvp, vmap

logger = logging.getLogger(__name__)

def X_and_dXdt_fd(net, x, sigma, labels, eps=1e-5):
    """
    Compute the output of the network and its derivative with respect to time t.
    First parameter of labels must be time t.
    Finite difference approximation to check correctness of jvp implementation.
    Parameters
    ----------
    net : callable
        The neural network model that takes inputs (x, sigma, t).
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    sigma : torch.Tensor
        Noise level tensor of shape (B,).
    labels : torch.Tensor
        labels tensor of shape (B, label_dim).
    eps : float, optional
        Finite difference step size, by default 1e-5.   

    Returns
    -------
    X : torch.Tensor
        Output of the network of shape (B, C, H, W).
    dXdt : torch.Tensor
        Derivative of the output with respect to time t, of shape (B, C, H, W).
    """
    lbl_p = labels.clone(); lbl_m = labels.clone()
    lbl_p[:, 0] += eps
    lbl_m[:, 0] -= eps
    up = net(x, sigma, lbl_p)
    um = net(x, sigma, lbl_m)
    u0 = net(x, sigma, labels)
    dudt_fd = (up - um) / (2*eps)
    return u0, dudt_fd


def X_and_dXdt(net, x, sigma, labels):
    """
    Compute the output of the network and its derivative with respect to time t.
    First parameter of labels must be time t.
    Parameters
    ----------
    net : callable
        The neural network model that takes inputs (x, sigma, t).
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    sigma : torch.Tensor
        Noise level tensor of shape (B,).
    labels : torch.Tensor
        labels tensor of shape (B, label_dim).

    Returns 
    -------
    X : torch.Tensor
        Output of the network of shape (B, C, H, W).
    dXdt : torch.Tensor
        Derivative of the output with respect to time t, of shape (B, C, H, W).
    """
    t0 = labels[:, 0]

    def f(t):
        lbl = labels.clone()
        lbl[:, 0] = t
        return net(x, sigma, lbl)
    
    X, dXdt = jvp(
        f,
        (t0,),
        (torch.ones_like(t0),)
    )
    return X, dXdt


def laplacian(u, dx):
    """
    Computes the Laplacian of a 2D tensor according to 
        ∇²u[i, j] ≈ (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4u[i, j]) / (dx^2)
    where dx is the grid spacing.
    For more details see
    https://en.wikipedia.org/wiki/Finite_difference_method#Example:_The_Laplace_operator

    Parameters
    ----------
    u : torch.Tensor
        Input tensor of shape (B, C, H, W).
    dx : float
        Grid spacing.

    Returns
    -------
    torch.Tensor
        Laplacian of u, of shape (B, C, H, W).
    """
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=u.dtype, device=u.device
    ).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 3, 3)
    
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode='reflect')  # shape (B, C, H+2, W+2)
    laplacian_u = torch.nn.functional.conv2d(u_padded, laplacian_kernel) / (dx ** 2)  # shape (B, C, H, W)
    return laplacian_u


def edm_sampler(
    net,            # EDMWrapper (calls Unet inside)
    device,         # device to run the sampler on  
    sample_shape,   # (B, C, H, W) shape of samples
    loss_fn,        # loss function to compute gradients
    loss_fn_kwargs, # extra args to pass to loss function
    labels,         # (B, label_dim) extra conditioning your Unet expects; use zeros if None
    zeta_a=1.0,     # weight for obs_a loss
    zeta_u=1.0,     # weight for obs_u loss
    zeta_pde=1.0,   # weight for pde loss
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    S_churn=0.0,
    S_min=0.0,
    S_max=float('inf'),
    S_noise=1.0,
    to_cpu=True,
    generator=None,
    debug=False,
    return_losses=False,
    compile_net=False,
):
    """
    Heun (2nd-order) EDM sampler, compatible with net(x, sigma, t).
    Returns a denoised sample at sigma=0 with shape (B, C, H, W).

    Parameters
    ----------
    net : callable
        The neural network model that takes inputs (x, sigma, t).
    device : torch.device
        Device to run the sampler on.
    sample_shape : tuple
        Shape of the samples to generate (B, C, H, W).
    loss_fn : callable
        Loss function to compute gradients, should match signature of heat_loss.
    loss_fn_kwargs : dict
        Extra arguments to pass to the loss function. Should include:
        - dx : float
            Spatial grid size.
        - obs_a : torch.Tensor
            (B, Ca, H, W) observations of initial condition.
        - obs_u : torch.Tensor
            (B, Cu, H, W) observation of solution at time T.
        - mask_a : torch.Tensor
            (B, Ca, H, W) binary mask for obs_a.
        - mask_u : torch.Tensor
            (B, Cu, H, W) binary mask for obs_u.
    labels : torch.Tensor
        (B, label_dim) extra conditioning your Unet expects; use zeros if None.
    zeta_a : float, optional
        Weight for obs_a loss, by default 1.0.
    zeta_u : float, optional
        Weight for obs_u loss, by default 1.0.
    zeta_pde : float, optional
        Weight for pde loss, by default 1.0.
    num_steps : int, optional
        Number of sampling steps, by default 18.
    sigma_min : float, optional
        Minimum noise level, by default 0.002.
    sigma_max : float, optional
        Maximum noise level, by default 80.0.
    rho : float, optional
        Rho parameter for noise schedule, by default 7.0.
    S_churn : float, optional
        Stochasticity parameter, by default 0.0.
    S_min : float, optional
        Minimum sigma for applying S_churn, by default 0.0.
    S_max : float, optional
        Maximum sigma for applying S_churn, by default float('inf').
    S_noise : float, optional
        Noise scale for S_churn, by default 1.0.
    debug : bool, optional
        If True, returns losses throughout sampling.

    returns
    -------
    torch.Tensor
        Denoised sample at sigma=0 with shape (B, C, H, W).
    np.ndarray (optional)
        Losses throughout sampling if debug=True, else None.
        """
    dtype_f = torch.float32     # net runs in fp32
    dtype_t = torch.float64     # keep time grid in fp64 for stability, as in EDM

    B = sample_shape[0]
    
    net.to(device=device)

    if compile_net:
        net = torch.compile(net)
        
    labels = labels.to(device=device, dtype=dtype_f)    # move labels to correct device and dtype

    if generator is None:
        generator = torch.Generator(device=device)

    # Initial sample at sigma_max
    latents = torch.randn(sample_shape, device=device, generator=generator)
    
    # Move loss function kwargs to correct device and dtype
    for key, val in loss_fn_kwargs.items():
        if isinstance(val, torch.Tensor):
            loss_fn_kwargs[key] = val.to(device=device, dtype=dtype_t)

    # Discretize sigmas per EDM (Karras et al. 2022), t_N = 0 appended.
    step_idx = torch.arange(num_steps, dtype=dtype_t, device=device)
    sigmas = (sigma_max**(1.0/rho) + step_idx/(num_steps-1) * (sigma_min**(1.0/rho) - sigma_max**(1.0/rho)))**rho
    sigmas = getattr(net, "round_sigma", lambda x: x)(sigmas)
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # length N+1, last = 0

    # Initialize x at sigma_0
    x_next = (latents.to(dtype_t) * sigmas[0])

    losses = torch.zeros((num_steps, 4), device=device)  # for debugging
    
    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):  # i = 0..N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True

        # Stochastic "churn" (optional)
        gamma = (min(S_churn / num_steps, (2.0**0.5) - 1.0)
                 if (sigma_cur >= S_min and sigma_cur <= S_max) else 0.0)
        sigma_hat = getattr(net, "round_sigma", lambda x: x)(sigma_cur + gamma * sigma_cur)

        # Add noise to increase sigma from sigma_cur to sigma_hat
        noise_scale = torch.sqrt(sigma_hat**2 - sigma_cur**2)
        x_hat = x_cur + (noise_scale * S_noise) * torch.randn_like(x_cur)

        # Euler step to t_next
        x_N, dxdt = X_and_dXdt_fd(net, x_hat.to(dtype_f), torch.full((B,), sigma_hat, device=device, dtype=dtype_f), labels)
        x_N, dxdt = x_N.to(dtype_t), dxdt.to(dtype_t)
        d_cur = (x_hat - x_N) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d_cur

        # Heun (2nd-order) correction unless final step
        if i < num_steps - 1:
            x_N, dxdt = X_and_dXdt_fd(net, x_next.to(dtype_f), torch.full((B,), sigma_next, device=device, dtype=dtype_f), labels)
            x_N, dxdt = x_N.to(dtype_t), dxdt.to(dtype_t)
            d_prime = (x_next - x_N) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Compute losses
        loss_pde, loss_obs_a, loss_obs_u = loss_fn(x_N, dxdt, **loss_fn_kwargs)

        if i <= 0.8 * num_steps:
            w_a, w_u, w_pde = zeta_a, zeta_u, zeta_pde
        else:
            w_a, w_u, w_pde = 0.1 * zeta_a, 0.1 * zeta_u, zeta_pde

        loss_comb = w_a * loss_obs_a + w_u * loss_obs_u + w_pde * loss_pde
        grad_x = torch.autograd.grad(loss_comb, x_cur, retain_graph=False)[0]
        x_next = x_next - grad_x

        losses[i] = torch.stack([loss_obs_a, loss_obs_u, loss_pde, loss_comb])
        if debug:
            print(f"iteration {i} complete")
 
    if debug:
        logger.info(f"Sampling completed with final losses - {loss_comb.item():.6f}")
    # Return at sigma=0 in fp32
    x = x_next.to(dtype_f).detach()
    if to_cpu:
        x = x.cpu()

    losses = losses.detach().cpu().numpy() if return_losses else None

    return x, losses