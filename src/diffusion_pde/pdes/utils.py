import h5py
import numpy as np

# deprecated
def save_dataset(
    filepath: str,
    A_train: np.ndarray,
    U_train: np.ndarray,
    labels_train: np.ndarray,
    A_test: np.ndarray,
    U_test: np.ndarray,
    labels_test: np.ndarray,
    t_steps: np.ndarray,
    T: float,
    dx: float,
    **attrs,
) -> None:
    """
    Save dataset to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the output HDF5 file.
    A_train : np.ndarray
        Initial conditions for training set, shape (N_train, ch_a, H, W).
    U_train : np.ndarray
        Solution data for training set, shape (N_train, ch_u, H, W, T).
    labels_train : np.ndarray
        Labels for training set, shape (N_train, label_dim).
    A_test : np.ndarray
        Initial conditions for test set, shape (N_test, ch_a, H, W).
    U_test : np.ndarray
        Solution data for test set, shape (N_test, ch_u, H, W, T).
    labels_test : np.ndarray
        Labels for test set, shape (N_test, label_dim).
    t_steps : np.ndarray
        Time steps array, shape (T,).
    T : float
        Total time duration.
    dx : float
        Spatial grid size.
    attrs : dict
        Additional attributes to store in the HDF5 file.

    Returns
    -------
    None
    """
    attrs["num_train"] = A_train.shape[0]
    attrs["num_test"] = A_test.shape[0]
    with h5py.File(filepath, "w") as f:
        grp_train = f.create_group("train")
        grp_test = f.create_group("test")
        grp_train.create_dataset("A", data=A_train)
        grp_train.create_dataset("U", data=U_train)
        grp_train.create_dataset("labels", data=labels_train)
        grp_test.create_dataset("A", data=A_test)
        grp_test.create_dataset("U", data=U_test)
        grp_test.create_dataset("labels", data=labels_test)
        f.create_dataset("t_steps", data=t_steps)

        f.attrs["T"] = T
        f.attrs["dx"] = dx

        for key, value in attrs.items():
            f.attrs[key] = value


def save_data(
    filepath: str,
    A: np.ndarray,
    U: np.ndarray,
    labels: np.ndarray,
    t_steps: np.ndarray,
    T: float,
    dx: float,
    dy: float,
    **attrs,
) -> None:
    """
    Save dataset to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the output HDF5 file.
    A_train : np.ndarray
        Initial conditions for training set, shape (N_train, ch_a, H, W).
    U_train : np.ndarray
        Solution data for training set, shape (N_train, ch_u, H, W, T).
    labels_train : np.ndarray
        Labels for training set, shape (N_train, label_dim).
    A_test : np.ndarray
        Initial conditions for test set, shape (N_test, ch_a, H, W).
    U_test : np.ndarray
        Solution data for test set, shape (N_test, ch_u, H, W, T).
    labels_test : np.ndarray
        Labels for test set, shape (N_test, label_dim).
    t_steps : np.ndarray
        Time steps array, shape (T,).
    T : float
        Total time duration.
    dx : float
        Spatial grid size in x direction.
    dy : float
        Spatial grid size in y direction.
    attrs : dict
        Additional attributes to store in the HDF5 file.

    Returns
    -------
    None
    """
    attrs["N"] = A.shape[0]
    with h5py.File(filepath, "w") as f:
        f.create_dataset("A", data=A)
        f.create_dataset("U", data=U)
        f.create_dataset("labels", data=labels)
        f.create_dataset("t_steps", data=t_steps)

        f.attrs["T"] = T
        f.attrs["dx"] = dx
        f.attrs["dy"] = dy

        for key, value in attrs.items():
            f.attrs[key] = value