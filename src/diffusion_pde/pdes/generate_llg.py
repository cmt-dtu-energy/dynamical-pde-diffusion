import os
import sys
from multiprocessing import Process, cpu_count
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from magtense.micromag import MicromagProblem
from tqdm import tqdm


def gen_s_state(
    res: tuple[int, int, int],
    grid_size: tuple[int, int, int],
    cuda: bool = False,
    show: bool = False,
) -> np.ndarray:
    mu0 = 4e-7 * np.pi

    problem_ini = MicromagProblem(
        res=res,
        grid_L=grid_size,
        m0=1 / np.sqrt(3),
        alpha=4.42e3,
        cuda=cuda,
    )

    h_ext = np.array([1, 1, 1]) / mu0

    def h_ext_fct(t) -> np.ndarray:
        return np.expand_dims(np.where(t < 1e-9, 1e-9 - t, 0), axis=1) * h_ext

    t_out, M_out = problem_ini.run_simulation(100e-9, 200, h_ext_fct, 2000)[:2]
    M_sq_ini = np.squeeze(M_out, axis=2)

    if show:
        plt.clf()
        plt.plot(t_out, np.mean(M_sq_ini[..., 0], axis=1), "rx")
        plt.plot(t_out, np.mean(M_sq_ini[..., 1], axis=1), "gx")
        plt.plot(t_out, np.mean(M_sq_ini[..., 2], axis=1), "bx")
        plt.show()

        plt.clf()
        plt.figure(figsize=(8, 2), dpi=80)
        s_state = np.reshape(M_sq_ini[-1], (res[1], res[0], 3))
        plt.quiver(s_state[..., 0], s_state[..., 1], pivot="mid")
        plt.show()

    return M_sq_ini[-1]


def gen_seq(
    m0_state: np.ndarray,
    res: list,
    grid_size: list,
    h_ext: np.ndarray = np.array((0, 0, 0)),
    t_steps: int = 500,
    t_per_step: float = 4e-12,
    cuda: bool = False,
    show: bool = False,
) -> np.ndarray:
    problem = MicromagProblem(
        res=res,
        grid_L=grid_size,
        m0=m0_state,
        alpha=4.42e3,
        gamma=2.21e5,
        A0=1.3e-11,
        Ms=8e5,
        K0=0.0,
        cuda=cuda,
    )

    t_end = t_per_step * t_steps
    # Rescale h_ext from mT to A/m
    h_ext = np.array(h_ext) / 1000 / (4 * np.pi * 1e-7)

    def h_ext_fct(t) -> np.ndarray:
        return np.expand_dims(t > -1, axis=1) * h_ext

    t_out, M_out = problem.run_simulation(t_end, t_steps + 1, h_ext_fct, 2000)[:2]
    M_sq = np.squeeze(M_out, axis=2)

    if show:
        plt.plot(t_out, np.mean(M_sq[..., 0], axis=1), "rx")
        plt.plot(t_out, np.mean(M_sq[..., 1], axis=1), "gx")
        plt.plot(t_out, np.mean(M_sq[..., 2], axis=1), "bx")
        plt.show()

    return M_sq


def db_std_prob_4(
    datapath: Path,
    n_seq: int,
    res: list,
    grid_size: tuple = (500e-9, 125e-9, 3e-9),
    t_steps: int = 500,
    t_per_step: float = 4e-12,
    h_ext_a: tuple = (0, 360),
    h_ext_n: tuple = (0, 50),
    seed: int = 0,
    intv: list | None = None,
    name: str | None = None,
    empty: bool = False,
    cuda: bool = False,
) -> tuple[str, int] | None:
    """
    Create a database of sequences with random external fields.
    """
    fname = name if name is not None else f"{n_seq}_{t_steps}_{res[0]}_{res[1]}"
    if intv is None:
        intv = [0, n_seq]
    if not empty:
        fname += f"_{intv[0]}_{intv[1]}"
    n_intv = intv[1] - intv[0]

    db = h5py.File(f"{datapath}/{fname}.h5", "w")
    db.create_dataset(
        "sequence", shape=(n_intv, t_steps, 3, res[0], res[1]), dtype="float32"
    )
    db.create_dataset("field", shape=(n_intv, 3), dtype="float32")
    if not empty:
        db.attrs["intv"] = intv

    if empty:
        # Suppress output and redirect stdout to /dev/null
        devnull = Path.open("/dev/null", "w")
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)

        ### SIMULATION WITH MAGTENSE ###
        s_state = gen_s_state(res, grid_size)

        # Suppress output and redirect stdout to /dev/null
        os.dup2(oldstdout_fno, 1)

        np.save(f"{datapath}/{res[0]}_{res[1]}_s_state.npy", s_state)
        db.attrs["res"] = res
        db.attrs["grid_size"] = grid_size
        db.attrs["t_steps"] = t_steps
        db.attrs["t_per_step"] = t_per_step
        db.attrs["h_ext_angle"] = h_ext_a
        db.attrs["h_ext_norm"] = h_ext_n
        db.attrs["seed"] = seed
        db.close()
        return fname, n_seq

    rng = np.random.default_rng(seed)
    rnd_mat = rng.random(size=(n_seq, 2))

    for i in tqdm(range(n_intv)):
        h_ext = np.zeros(3)
        d = (h_ext_n[1] - h_ext_n[0]) * rnd_mat[i, 0] + h_ext_n[0]
        theta = np.deg2rad((h_ext_a[1] - h_ext_a[0]) * rnd_mat[i, 1] + h_ext_a[0])
        h_ext[:2] = d * np.array([np.cos(theta), np.sin(theta)])
        # Stored in mT
        db["field"][i] = h_ext

        # Suppress output and redirect stdout to /dev/null
        devnull = Path.open("/dev/null", "w")
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)

        ### SIMULATION WITH MAGTENSE ###
        seq = gen_seq(
            m0_state=np.load(f"{datapath}/{res[0]}_{res[1]}_s_state.npy"),
            res=res,
            grid_size=grid_size,
            h_ext=h_ext,
            t_steps=t_steps,
            t_per_step=t_per_step,
            cuda=cuda,
        )

        # Suppress output and redirect stdout to /dev/null
        os.dup2(oldstdout_fno, 1)

        # Output shape: (t, res_x, res_y, 3)
        db["sequence"][i] = np.moveaxis(
            seq[:t_steps].copy().reshape(t_steps, res[1], res[0], 3).swapaxes(1, 2), -1, 1
        )

    db.close()


def create_db_mp(
    n_workers: int | None = None,
    datapath: Path | None = None,
    **kwargs,
) -> None:
    if datapath is None:
        datapath = Path(__file__).parent / ".." / ".." / ".." / "data"
    if not datapath.exists():
        datapath.mkdir(parents=True)
    kwargs["datapath"] = datapath

    db_name, n_tasks = db_std_prob_4(**kwargs, empty=True)

    if n_workers is None:
        n_workers = cpu_count()
    intv = n_tasks // n_workers
    if n_tasks % n_workers > 0:
        intv += 1

    l_p = []
    for i in range(n_workers):
        end_intv = min((i + 1) * intv, n_tasks)
        kwargs["intv"] = [i * intv, end_intv]
        p = Process(target=db_std_prob_4, kwargs=kwargs)
        p.daemon = True
        p.start()
        l_p.append(p)
        if end_intv == n_tasks:
            break

    try:
        for p in l_p:
            p.join()

    except KeyboardInterrupt:
        for p in l_p:
            p.terminate()

        path = datapath.glob("**/*")
        fnames = [
            x.name
            for x in path
            if x.is_file()
            and x.name[: len(db_name)] == db_name
            and x.name[:-3] != db_name
        ]

        for name in fnames:
            Path(datapath, name).unlink()

        Path(datapath, f"{db_name}.h5").unlink()
        sys.exit(130)

    path = datapath.glob("**/*")
    fnames = [
        x.name
        for x in path
        if x.is_file() and x.name[: len(db_name)] == db_name and x.name[:-3] != db_name
    ]

    with h5py.File(f"{datapath}/{db_name}.h5", mode="a") as db_t:
        for name in fnames:
            print(Path(datapath, name))
            with h5py.File(Path(datapath, name), mode="r") as db_s:
                intv = db_s.attrs["intv"]
                for key in db_s:
                    db_t[key][intv[0] : intv[1]] = db_s[key]
            Path(datapath, name).unlink()

    print("Database created")


if __name__ == "__main__":
    db_kwargs = {
        "res": [16, 4, 1],
        "grid_size": [500e-9, 125e-9, 3e-9],
        "cuda": True,
        "n_seq": 4,
    }

    create_db_mp(n_workers=1, **db_kwargs)
