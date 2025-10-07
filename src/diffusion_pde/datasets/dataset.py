import torch
import numpy as np
from typing import Optional

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
        init_state: np.ndarray, 
        data: np.ndarray, 
        t_steps: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__()
        # assume init_state is (N, ch_a, h, w)
        # assume data is (N, ch_u, h, w, t)
        assert len(init_state.shape) == 4, f"Dimensions of 'init_state' should be (N, ch_a, h, w) but got {init_state.shape}"
        assert len(data.shape) == 5, f"Dimensions of 'data' should be (N, ch_u, h, w, t) but got {data.shape}"
        assert init_state.shape[0] == data.shape[0], f"Number of initial points and trajectories must match but got {init_state.shape[0]} and {data.shape[0]}"

        self.init_state = torch.from_numpy(init_state).float()
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.t_steps = torch.from_numpy(t_steps).float()

        self.N, self.T = data.shape[0], data.shape[-1]

        self.g = generator

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        # sample random timestep
        t_idx = torch.randint(0, self.T, (1,), generator=self.g).item()

        # slice data snapshot at timestep
        snap = self.data[idx, ..., t_idx] # shape (ch_u, h, w)
        X = torch.cat((self.init_state[idx], snap), dim=0)  # shape (ch_a + ch_u, h, w)
        # get corresponding time value
        label = self.t_steps[t_idx]
        if self.labels is not None:
            label = torch.cat((torch.tensor([label]), self.labels[idx]), dim=0)
        return X, label