import torch
import numpy as np
import h5py
from pathlib import Path
from diffusion_pde.utils import get_repo_root

class NoTimeDataset(torch.utils.data.Dataset):
    """
    Dataset for heat equation without conditioning.

    Parameters
    ----------
    data : np.ndarray
        Numpy array of shape (N, S, S, 2) containing the dataset
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None
    ):
        self.data = torch.tensor(data).float()  # (N, C, S, S, 2)
        self.labels = torch.tensor(labels).float() if labels is not None else None
        self.N = data.shape[0]

    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor | None]:

        X = torch.cat((self.data[idx, ..., 0], self.data[idx, ..., -1]), dim=0)  # (2*C, S, S)
        labels = self.labels[idx] if self.labels is not None else None
        return {"X": X, "labels": labels}
    

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
        labels: np.ndarray | None = None,
        start_at_t0: bool = True,
        generator: torch.Generator | None = None
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

    def __getitem__(self, idx) -> dict[str, torch.Tensor | None]:
        
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
        labels: np.ndarray | None = None,
        start_at_t0: bool = False,
        generator: torch.Generator | None = None
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

    def __getitem__(self, idx) -> dict[str, torch.Tensor | None]:
        
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



class ValidationDataset(torch.utils.data.Dataset):
    """
    Validation dataset for diffusion models.
    Each item is a dictionary containing:
        - "A": Initial state tensor of shape (C, H, W)
        - "U": State tensor at target time of shape (C, H, W)
        - "labels": Optional labels tensor of shape (label_dim,)
    
    Parameters
    ----------
    data : np.ndarray
        Numpy array of shape (N, C, H, W, T) containing the dataset
    t_steps : np.ndarray
        1D numpy array of shape (T,) containing the time steps
    labels : Optional[np.ndarray], optional
        Optional numpy array of shape (N, label_dim) containing additional labels, by default None. 
    time_as_label : bool, optional
        Whether to include time as part of the labels, by default False.
    include_t0_as_target : bool, optional
        Whether to include the initial time step as part of the target U, by default False.  
    """
    def __init__(
        self, 
        data: np.ndarray, 
        t_steps: np.ndarray, 
        labels: np.ndarray | None = None, 
        time_as_label: bool = False,
        include_t0_as_target: bool = False,
    ) -> None:
        data = torch.tensor(data).float()  # (N, C, H, W, T)
        t_steps = torch.tensor(t_steps).float()  # (T,)
        labels = torch.tensor(labels).float() if labels is not None else None

        N, C, H, W, T = data.shape

        if len(t_steps) != T:
            raise ValueError(f"Length of t_steps ({len(t_steps)}) must match the last dimension of data ({T})")
        if len(t_steps) < 2:
            raise ValueError(f"t_steps must contain at least 2 time steps, but got {len(t_steps)}")
        
        T = T if include_t0_as_target else T - 1
        t_idx_start = 0 if include_t0_as_target else 1

        A = data[..., 0].repeat_interleave(T, dim=0)  # (N, C, H, W) -> (N*T, C, H, W)
        U = data[..., t_idx_start:].permute(0, 4, 1, 2, 3).reshape(N * T, C, H, W) # (N, C, H, W, T) -> (N*T, C, H, W)

        self.data = torch.cat((A, U), dim=1)  # (N*T, 2*C, H, W)

        self.labels = None
        if labels is not None:
            if labels.ndim == 1:
                labels = labels.reshape((-1, 1))  # ensure labels is (N, label_dim):
            labels = labels.repeat_interleave(T, dim=0)  # (N*T, label_dim)
            if time_as_label:
                t_steps_expanded = t_steps[t_idx_start:].repeat(N)  # (N*T,)
                self.labels = torch.cat((t_steps_expanded.unsqueeze(1), labels), dim=1)  # (N*T, 1 + label_dim)
            else:
                self.labels = labels  # (N*T, label_dim)

        self.N = N * T
        self.C = C

    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor | None]:
        A = self.data[idx, :self.C, ...]  # (C, H, W)
        U = self.data[idx, self.C:, ...]  # (C, H, W)
        labels = self.labels[idx] if self.labels is not None else None
        return {"A": A, "U": U, "labels": labels}


def collate_optional(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if batch[0][key] is not None:
            batch_dict[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            batch_dict[key] = None
    return batch_dict


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

    if "no_cond" in cfg.dataset.data.name.lower() or "no_time" in cfg.dataset.data.name.lower():
        dataset = NoTimeDataset(data[train_idxs, ...], labels=labels[train_idxs] if labels is not None else None)
        valset = NoTimeDataset(data[val_idxs, ...], labels=labels[val_idxs] if labels is not None else None)
    elif method == "forward":
        dataset = DiffusionDatasetForward(data[train_idxs, ...], t_steps, labels=labels[train_idxs] if labels is not None else None, start_at_t0=start_at_t0)
        valset = DiffusionDatasetForward(data[val_idxs, ...], t_steps, labels=labels[val_idxs] if labels is not None else None, start_at_t0=start_at_t0)
    else:
        dataset = DiffusionDataset(data[train_idxs, ...], t_steps, labels=labels[train_idxs] if labels is not None else None, start_at_t0=start_at_t0)
        valset = DiffusionDataset(data[val_idxs, ...], t_steps, labels=labels[val_idxs] if labels is not None else None, start_at_t0=start_at_t0)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_optional)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate_optional)
    return dataloader, valloader


def get_validation_dataloader(data_path: Path | str, time_as_label: bool, include_t0_as_target: bool) -> torch.utils.data.DataLoader:
    """
    Utility function to load validation dataset from .h5 file and create a dataloader.

    Parameters
    ----------
    data_path : str
        Path to the .h5 file containing the dataset.
    time_as_label : bool
        Whether to include time as part of the labels.
    include_t0_as_target : bool
        Whether to include the initial time step as part of the target U.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    """
    datapath = Path(data_path)
    repository_root = get_repo_root()
    if not datapath.is_absolute():
        datapath = repository_root / datapath

    with h5py.File(datapath, "r") as f:
        data = f["U"][:]        # (N, ch_u, h, w, T)
        t_steps = f["t_steps"][:]     # (T,)
        labels = f["labels"][:] if "labels" in f else None  # (N,) or (N, label_dim)

    valset = ValidationDataset(data, t_steps, labels=labels, time_as_label=time_as_label, include_t0_as_target=include_t0_as_target)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, collate_fn=collate_optional)
    return valloader
