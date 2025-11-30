import torch
import numpy as np
import h5py
from typing import Optional
from pathlib import Path
from diffusion_pde.utils import get_repo_root

class DiffusionDataset(torch.utils.data.Dataset):
    """
    Diffusion dataset compatible with torch DataLoader.
    Each item is a tuple (X, label) where X is the concatenation of the
    initial state and a snapshot of the trajectory at time t. the label
    is a vector containing the time t and any additional labels.
    Note that t is sampled randomly for each item from the t_steps provided.

    Parameters
    ----------
    init_state : torch.Tensor
        Tensor of shape (N, ch_a, h, w) representing the initial states.
    data : torch.Tensor
        Tensor of shape (N, ch_u, h, w, T) representing the trajectories.
    t_steps : torch.Tensor
        1D tensor of shape (T,) representing the time steps corresponding to the last dimension of `data`.
    labels : Optional[torch.Tensor], optional
        Optional tensor of shape (N, label_dim) representing additional labels, by default None.
    generator : Optional[torch.Generator], optional
        Random number generator for reproducibility, by default None.
    """
    def __init__(self,
        data: np.ndarray, 
        t_steps: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        start_at_t0: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__()
        # assume data is (N, ch_u, h, w, t)
        assert len(data.shape) == 5, f"Dimensions of 'data' should be (N, ch_u, h, w, t) but got {data.shape}"
        self.start_at_t0 = start_at_t0
        
        self.data = torch.tensor(data).float()
        self.labels = torch.tensor(labels).float() if labels is not None else None
        if self.labels is not None and self.labels.ndim == 1:
            self.labels = self.labels.reshape((-1, 1))  # ensure labels is (N, label_dim):
        self.t_steps = torch.tensor(t_steps).float()

        self.N, self.T = data.shape[0], data.shape[-1]
        self.g = generator

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        # sample random timestep
        t0_idx = 0
        if not self.start_at_t0:
            t0_idx = torch.randint(0, self.T, (1,), generator=self.g).item()

        tf_idx = torch.randint(t0_idx, self.T, (1,), generator=self.g).item()

        # slice data snapshot at timestep
        snap_t0 = self.data[idx, ..., t0_idx]  # shape (ch_u, h, w)
        snap_tf = self.data[idx, ..., tf_idx] # shape (ch_u, h, w)
        X = torch.cat((snap_t0, snap_tf), dim=0)  # shape (ch_a + ch_u, h, w)
        # get corresponding time value
        tau = self.t_steps[tf_idx] - self.t_steps[t0_idx]

        if self.labels is not None:
            label = torch.cat((torch.tensor([tau]), self.labels[idx]), dim=0)

        return {"X": X, "labels": label}
    

class DiffusionDatasetForward(torch.utils.data.Dataset):
    """
    Diffusion dataset compatible with torch DataLoader.
    Each item is a tuple (X, label) where X is the concatenation of the
    initial state and a snapshot of the trajectory at time t. the label
    is a vector containing the time t and any additional labels.
    Note that t is sampled randomly for each item from the t_steps provided.

    Parameters
    ----------
    data : torch.Tensor
        Tensor of shape (N, ch_u, h, w, T) representing the trajectories.
    t_steps : torch.Tensor
        1D tensor of shape (T,) representing the time steps corresponding to the last dimension of `data`.
    labels : Optional[torch.Tensor], optional
        Optional tensor of shape (N, label_dim) representing additional labels, by default None.
    generator : Optional[torch.Generator], optional
        Random number generator for reproducibility, by default None.
    """
    def __init__(self,
        data: np.ndarray, 
        t_steps: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        obs: Optional[np.ndarray] = None,
        start_at_t0: bool = False,
        generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__()
        # assume data is (N, ch_u, h, w, t)
        assert len(data.shape) == 5, f"Dimensions of 'data' should be (N, ch_u, h, w, t) but got {data.shape}"

        self.start_at_t0 = start_at_t0

        self.data = torch.tensor(data).float()
        self.labels = torch.tensor(labels).float() if labels is not None else None
        if self.labels is not None and self.labels.ndim == 1:
            self.labels = self.labels.reshape((-1, 1))  # ensure labels is (N, label_dim):
        self.obs = torch.tensor(obs).float() if obs is not None else None
        self.t_steps = torch.tensor(t_steps).float()
        self.N, self.T = data.shape[0], data.shape[-1]

        self.g = generator

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        # sample random timestep
        t0_idx = 0
        if not self.start_at_t0:
            t0_idx = torch.randint(0, self.T, (1,), generator=self.g).item()

        tf_idx = torch.randint(t0_idx, self.T, (1,), generator=self.g).item()

        # slice data snapshot at timestep
        # get corresponding time value
        obs = self.data[idx, ..., t0_idx]  # shape (ch_u, h, w, 2)
        X = self.data[idx, ..., tf_idx] # shape (ch_u, h, w)

        tau = self.t_steps[tf_idx] - self.t_steps[t0_idx]

        if self.labels is not None:
            label = torch.cat((torch.tensor([tau]), self.labels[idx]), dim=0)

        return {"obs": obs, "X": X, "labels": label}
    
    
def get_dataloaders(cfg):
    """
    Utility function to load dataset from .h5 file and create a dataloader.

    Parameters
    ----------
    datapath : str
        Path to the .h5 file containing the dataset.
    batch_size : int
        Batch size for the dataloader.
    shuffle : bool, optional
        Whether to shuffle the dataset, by default True.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader for the dataset.
    """
    datapath = Path(cfg.dataset.data.datapath)
    repository_root = get_repo_root()
    if not datapath.is_absolute():
        datapath = repository_root / datapath

    method = cfg.dataset.method
    start_at_t0 = cfg.dataset.start_at_t0
    batch_size = cfg.dataset.training.batch_size
    shuffle = cfg.dataset.training.shuffle
    val_percent = cfg.dataset.training.val_percent

    with h5py.File(datapath, "r") as f:
        data = f["U"][:]        # (N, ch_u, h, w, T)
        t_steps = f["t_steps"][:]     # (T,)
        labels = f["labels"][:] if "labels" in f else None  # (N,) or (N, label_dim)

    N = data.shape[0]
    val_size = int(N * val_percent)
    train_size = N - val_size
    
    idxs = torch.arange(N)
    if shuffle:
        idxs = idxs[torch.randperm(N)]
    train_idxs, val_idxs = idxs[:train_size], idxs[train_size:]

    if method == "forward":
        dataset = DiffusionDatasetForward(data[train_idxs, ...], t_steps, labels=labels[train_idxs] if labels is not None else None, start_at_t0=start_at_t0)
        valset = DiffusionDatasetForward(data[val_idxs, ...], t_steps, labels=labels[val_idxs] if labels is not None else None, start_at_t0=start_at_t0)
    else:
        dataset = DiffusionDataset(data[train_idxs, ...], t_steps, labels=labels[train_idxs] if labels is not None else None, start_at_t0=start_at_t0)
        valset = DiffusionDataset(data[val_idxs, ...], t_steps, labels=labels[val_idxs] if labels is not None else None, start_at_t0=start_at_t0)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return dataloader, valloader