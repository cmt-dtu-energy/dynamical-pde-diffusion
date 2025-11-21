import torch
import functools

PDE_LOSS_REGISTRY = {}

def register(name=None):
    """Decorator to register a function in REGISTRY."""
    def decorator(func):
        key = name or func.__name__
        if key in PDE_LOSS_REGISTRY:
            raise ValueError(f"Key '{key}' already in REGISTRY")
        PDE_LOSS_REGISTRY[key] = func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def X_and_dXdt_fd(net, x, sigma, labels, eps=1e-5, **kwargs):
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
    up = net(x, sigma, lbl_p, **kwargs)
    um = net(x, sigma, lbl_m, **kwargs)
    u0 = net(x, sigma, labels, **kwargs)
    dudt_fd = (up - um) / (2*eps)

    del lbl_p, lbl_m, up, um        # free up memory

    return u0, dudt_fd

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

@register(name="heat")
def heat_pde_loss(u, dudt, labels, dx=1.0):
    alpha = labels[:, 1].view(-1, 1, 1, 1)
    laplacian_u = laplacian(u, dx)
    loss_pde = torch.mean((dudt - alpha * laplacian_u) ** 2, dim=(1,2,3))
    return loss_pde