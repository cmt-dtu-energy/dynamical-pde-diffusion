import h5py
import numpy as np

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