"""
PyTorch DataLoader and Lightning DataModule for tokenizer training.

Provides windowed ``Dataset`` that plugs into :class:`EphysDataModule`. A
backend-agnostic :class:`WindowedSession` base handles windowing and per-session
standardisation; backends supply the data source:

- :class:`H5Session` / :func:`build_h5_dataset`: a ``"data"`` array of shape
  ``(n_samples, n_channels)`` per session, sliced lazily from a h5 file.
- :class:`FIFSession` / :func:`build_fif_dataset`: parcellated MEG FIFs read
  via MNE-Python
"""

from __future__ import annotations

import h5py
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import random
import torch

from pathlib import Path
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    Subset,
    get_worker_info,
)
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _default_worker_init_fn(worker_id: int) -> None:
    """
    Default worker initialization function for DataLoader workers
    to ensure different but reproducible random seeds for each worker.

    - Uses torch.initial_seed() (set by DataLoader) to derive worker
      seeds and then seeds numpy, python random, and torch for safety.
    - Ensures deterministic but distinct RNG state per worker.

    Parameters
    ----------
    worker_id : int
        The ID of the worker process.
    """
    # Get worker info
    worker_info = get_worker_info()
    if worker_info is None:
        return

    # Set seed for numpy, random, and torch
    seed = int(torch.initial_seed() % (2**32)) + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _collate_default(batch: Sequence[dict]) -> Dict[str, Any]:
    """
    Default collate function for session-window items.

    Each item is expected to be:
        {"data": np.ndarray (L, C), "times": np.ndarray (L,), "info": dict}
    where L is the sequence length and C is the number of channels.

    Parameters
    ----------
    batch : List[Dict[str, Union[np.ndarray, dict]]]
        A list of items to collate.

    Returns
    -------
    batch_result : Dict[str, Union[Tensor, List]]
        A dictionary containing the collated data.
    """
    # Get items in batch
    datas = [np.array(item["data"], dtype=np.float32) for item in batch]
    times = [np.array(item["times"], dtype=np.float32) for item in batch]
    infos = [item.get("info", {}) for item in batch]  # keep original info per sample

    # Stack along the batch dimension
    batch_data = torch.from_numpy(np.stack(datas, axis=0))  # shape: (B, L, C)
    batch_times = torch.from_numpy(np.stack(times, axis=0))  # shape: (B, L)

    return {"data": batch_data, "times": batch_times, "info": infos}


def _make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: Optional[int] = 4,
    pin_memory: Optional[bool] = True,
    persistent_workers: Optional[bool] = True,
    drop_last: Optional[bool] = False,
    sampler: Optional[Sampler] = None,
    collate_fn=_collate_default,
) -> DataLoader:
    """
    Creates a DataLoader for a given dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load.
    batch_size : int
        The number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset.
        If `sampler` is provided, `shuffle` is ignored (sampler takes precedence).
    num_workers : int
        The number of worker processes to use for data loading.
    pin_memory : bool
        Whether to pin memory for the DataLoader.
    persistent_workers : bool
        Whether to use persistent workers for the DataLoader.
    drop_last : bool
        Whether to drop the last incomplete batch.
    sampler : Optional[Sampler]
        A sampler to use for sampling the dataset.
    collate_fn : callable
        A function to collate samples into batches.

    Returns
    -------
    dataloader: DataLoader
        A DataLoader for a given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),  # shuffle only if no sampler is provided
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=_default_worker_init_fn,
    )


class EphysDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for continuous session-based datasets.

    Expects the input dataset to wrap a ``ConcatDataset`` of per-session
    sub-datasets (accessible via ``dataset.dataset``). Each sub-dataset must
    expose a ``.subject`` attribute if subject-level splitting is used.

    Parameters
    ----------
    dataset : Dataset
        The PyTorch dataset.
    batch_size : int
        Batch size for training/validation.
    val_split : float
        Fraction in (0,1) used for validation split.
    split_method : str
        - "subject": allocate whole subjects to train/val
        - "window": random windows across whole concatenation
        - "subject_window": per-subject window-level split (train/val within each subject)
    is_distributed : bool
        Whether training will run under DDP (Distributed Data Parallel).
        If True, enables DistributedSampler.
    seed : int
        RNG seed for reproducibility.
    num_workers : int
        Number of worker processes for data loading.
    pin_memory : bool
        Whether to pin memory for the DataLoader.
    persistent_workers : bool
        Whether to use persistent workers for the DataLoader.
    drop_last : bool
        Whether to drop the last incomplete batch in training dataloader.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        val_split: float,
        split_method: Optional[str] = "subject",
        is_distributed: Optional[bool] = False,
        seed: Optional[int] = 42,
        num_workers: Optional[int] = 4,
        pin_memory: Optional[bool] = True,
        persistent_workers: Optional[bool] = True,
        drop_last: Optional[bool] = True,
    ):
        super().__init__()

        if not isinstance(dataset.dataset, ConcatDataset):
            raise TypeError("Expects an input dataset to include a ConcatDataset.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = float(val_split)
        self.split_method = split_method
        self.is_distributed = is_distributed
        self.seed = int(seed)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = bool(drop_last)

        # Set placeholders
        self.train_idx: Optional[List[int]] = None
        self.val_idx: Optional[List[int]] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    # ----------------
    # Helper Functions
    # ----------------

    def _subject_index_ranges(self):
        """
        Returns index ranges for each subject in the dataset.

        Note that the output relies on ConcatDataset.datasets being in
        the same order as the subjects you used to build the dataset.

        Returns
        -------
        ranges : List[Tuple[str, int, int]]
            A list of (subject_id, start_index, end_index) tuples.
            End index is exclusive.
        """
        ranges: List[Tuple[str, int, int]] = []
        offset = 0
        for ds in self.dataset.dataset.datasets:
            length = len(ds)  # number of sequences
            subj_id = getattr(ds, "subject", None)
            ranges.append((subj_id, offset, offset + length))
            offset += length
        return ranges

    # ----------------
    # Splitting Logics
    # ----------------

    def _split_by_subjects(self) -> Tuple[List[int], List[int]]:
        """
        Splits the dataset into training and validation sets based on subjects.

        Returns
        -------
        train_idx : List[int]
            The subject indices for the training set.
        val_idx : List[int]
            The subject indices for the validation set.
        """
        ranges = self._subject_index_ranges()
        subject_ids = [r[0] for r in ranges]
        if any(s is None for s in subject_ids):
            raise RuntimeError(
                "All datasets in the ConcatDataset must have a 'subject' attribute for subject-based splitting."
            )

        # Shuffle subjects (deterministic)
        rng = random.Random(self.seed)
        ids_perm = subject_ids.copy()
        rng.shuffle(ids_perm)

        # Get number of train and validation instances
        n_subjects = len(ids_perm)
        n_val = int(np.floor(self.val_split * n_subjects))
        n_train = n_subjects - n_val
        if n_train <= 0:
            raise ValueError("Not enough subjects for the requested splits.")

        # Split into train/val
        train_idx = ids_perm[:n_train]
        val_idx = ids_perm[n_train:n_train + n_val]

        # Map subjects to index ranges
        subj_to_range = {s: (start, end) for s, start, end in ranges}
        def _ranges_to_idxlist(subj_list: Iterable[str]) -> List[int]:
            idxs = []
            for s in subj_list:
                start, end = subj_to_range[s]
                idxs.extend(list(range(start, end)))
            return idxs

        return _ranges_to_idxlist(train_idx), _ranges_to_idxlist(val_idx)

    def _split_by_windows(self) -> Tuple[List[int], List[int]]:
        """
        Splits the dataset into training and validation sets based on windows
        (i.e., sequences).

        Returns
        -------
        train_idx : List[int]
            The sequence indices for the training set.
        val_idx : List[int]
            The sequence indices for the validation set.
        """
        # Get number of train and validation instances
        n_total_windows = len(self.dataset)
        n_val = int(np.floor(self.val_split * n_total_windows))
        n_train = n_total_windows - n_val
        if n_train <= 0:
            raise ValueError("Not enough windows for the requested splits.")

        # Shuffle windows
        rng = random.Random(self.seed)
        all_idx = list(range(n_total_windows))
        rng.shuffle(all_idx)

        # Split into train/val
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:n_train + n_val]

        return train_idx, val_idx

    def _split_by_subject_windows(self) -> Tuple[List[int], List[int]]:
        """
        Splits the dataset into training and validation sets based on windows
        per each subject.

        Returns
        -------
        train_idx : List[int]
            The sequence indices for the training set.
        val_idx : List[int]
            The sequence indices for the validation set.
        """
        # Set random seed
        rng = random.Random(self.seed)

        train_idx, val_idx = [], []
        ranges = self._subject_index_ranges()
        for s, start, end in ranges:
            # Shuffle window for each subject (deterministic)
            indices = list(range(start, end))
            rng.shuffle(indices)

            # Get number of train and validation instances
            n_windows = len(indices)
            n_val = int(np.floor(self.val_split * n_windows))
            n_train = n_windows - n_val
            if n_train <= 0:
                raise ValueError(f"Subject {s} has not enough windows for the requested splits.")

            # Split into train/val for each subject
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])

        return train_idx, val_idx

    # ------------------------------
    # LightningDataModule Interfaces
    # ------------------------------

    def prepare_data(self) -> None:
        # If we need to predownload or validate the cache files, we can do it here.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the training and validation datasets by splitting the indices.
        Called on every process in DDP.

        Parameters
        ----------
        stage : Optional[str]
            The stage to set up (train/val/test). If None, sets up all stages.
        """
        # Split train and validation indices
        if self.split_method == "subject":
            train_idx, val_idx = self._split_by_subjects()
        elif self.split_method == "window":
            train_idx, val_idx = self._split_by_windows()
        elif self.split_method == "subject_window":
            train_idx, val_idx = self._split_by_subject_windows()
        else:
            raise ValueError(f"Invalid split method: {self.split_method}")

        # Convert to sorted lists
        self.train_idx = sorted(train_idx)
        self.val_idx = sorted(val_idx)

        # Prepare Subsets for eager construction
        self.train_dataset = Subset(self.dataset, self.train_idx)
        self.val_dataset = Subset(self.dataset, self.val_idx)

    def _get_sampler(self, dataset: Dataset, shuffle: bool) -> Optional[Sampler]:
        """
        Returns a DistributedSampler in distributed mode; otherwise, returns None
        (DataLoader shuffle handles it automatically).

        Parameters
        ----------
        dataset : Dataset
            The dataset to sample from.
        shuffle : bool
            Whether to shuffle the data samples.

        Returns
        -------
        sampler : Optional[Sampler]
            A DistributedSampler if in distributed mode; otherwise, None.
        """
        if self.is_distributed:
            return DistributedSampler(dataset, shuffle=shuffle, seed=self.seed)
        return None

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.
        """
        sampler = self._get_sampler(self.train_dataset, shuffle=True)
        return _make_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),  # shuffle only if no sampler is provided
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.
        """
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return []

        sampler = self._get_sampler(self.val_dataset, shuffle=False)
        return _make_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            sampler=sampler,
        )

    def full_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the full dataset.
        Useful helper for evaluation or inference over the full dataset.
        """
        return _make_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )


class WindowedSession(Dataset):
    """
    Non-overlapping windowed view of one session.

    Fixed-length non-overlapping windows, optional per-channel per-session
    z-score standardisation.

    Subclasses supply the data source via three hooks:

    - ``_load_array()`` -> ``(n_samples, n_channels)`` array, read once at init
      to determine the length and standardisation statistics.
    - ``_open()`` -> a per-process resource (e.g. an open file handle or an
      in-memory array) that is cached and passed to ``_read_window``.
    - ``_read_window(resource, start, end)`` -> the ``(window_len, n_channels)``
      slice for one window.

    Parameters
    ----------
    window_len : int
        Window length in samples; windows are non-overlapping. Any trailing
        samples shorter than ``window_len`` are dropped.
    sfreq : float
        Sample rate in Hz, used only to populate the ``times`` array.
    info : dict
        Metadata dict copied into every item's ``info`` field. Must contain a
        ``"subject"`` key if subject-level splitting will be used downstream.
    standardize : bool
        If True, apply per-channel, per-session z-score standardisation
        (subtract mean, divide by std; zero stds are treated as 1).
    """

    def __init__(
        self,
        window_len: int,
        sfreq: float,
        info: Dict[str, Any],
        standardize: bool = True,
    ):
        self.window_len = int(window_len)
        self.sfreq = float(sfreq)
        self.info = dict(info)
        self.standardize = bool(standardize)

        # Required by EphysDataModule subject-splitting logic
        self.subject = self.info.get("subject")

        arr = self._load_array()
        self.n_samples = int(arr.shape[0])
        self.n_channels = int(arr.shape[1])
        if self.standardize:
            data = arr.astype(np.float64)
            self._mean = data.mean(axis=0).astype(np.float32)
            std = data.std(axis=0).astype(np.float32)
            std[std == 0.0] = 1.0
            self._std = std
        else:
            self._mean = None
            self._std = None

        self.n_windows = self.n_samples // self.window_len
        # Opened lazily per process; re-opened if the PID changes so a resource
        # opened in the main process isn't reused (and silently desynced) by a
        # forked/spawned DataLoader worker.
        self._resource: Optional[Any] = None
        self._resource_pid: Optional[int] = None

    # backend hooks
    def _load_array(self) -> np.ndarray:
        raise NotImplementedError

    def _open(self) -> Any:
        raise NotImplementedError

    def _read_window(self, resource: Any, start: int, end: int) -> np.ndarray:
        raise NotImplementedError

    # shared behaviour
    def __len__(self) -> int:
        return self.n_windows

    def __getstate__(self) -> Dict[str, Any]:
        # Strip the per-process resource so the dataset pickles cleanly into
        # workers (spawn multiprocessing context).
        state = self.__dict__.copy()
        state["_resource"] = None
        state["_resource_pid"] = None
        return state

    def _handle(self) -> Any:
        pid = os.getpid()
        if self._resource is None or self._resource_pid != pid:
            self._resource = self._open()
            self._resource_pid = pid
        return self._resource

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(idx)
        start = idx * self.window_len
        end = start + self.window_len
        data = self._read_window(self._handle(), start, end).astype(
            np.float32, copy=False
        )
        if self.standardize:
            data = (data - self._mean) / self._std
        times = np.arange(start, end, dtype=np.float32) / self.sfreq
        return {"data": data, "times": times, "info": self.info}


class H5Session(WindowedSession):
    """
    h5-backed windowed session.

    Reads a single ``"data"`` dataset of shape ``(n_samples, n_channels)`` and
    slices windows lazily from the open file, so no full copy is held in memory.

    Parameters
    ----------
    h5_path : str
        Path to a session h5 file containing a ``"data"`` dataset.
    window_len, sfreq, info, standardize
        See :class:`WindowedSession`.
    """

    def __init__(
        self,
        h5_path: str,
        window_len: int,
        sfreq: float,
        info: Dict[str, Any],
        standardize: bool = True,
    ):
        self.h5_path = str(h5_path)
        super().__init__(window_len, sfreq, info, standardize)

    def _load_array(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            return f["data"][...]

    def _open(self) -> h5py.File:
        return h5py.File(self.h5_path, "r")

    def _read_window(self, resource: h5py.File, start: int, end: int) -> np.ndarray:
        return resource["data"][start:end, :]


class SessionDataset(Dataset):
    """
    Concatenates windowed sessions of any backend (:class:`WindowedSession`).

    ``.dataset`` is exposed as a ``ConcatDataset`` so it can be used with
    :class:`EphysDataModule`.
    """

    def __init__(self, sessions: Sequence[WindowedSession]):
        if len(sessions) == 0:
            raise ValueError("SessionDataset requires at least one session.")
        self.dataset = ConcatDataset(list(sessions))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]


def build_h5_dataset(
    sessions_csv: str,
    h5_dir: str,
    window_len: int,
    sfreq: float = 250.0,
    standardize: bool = True,
    include_sessions: Optional[Sequence[str]] = None,
    info_cols: Sequence[str] = (
        "session",
        "dataset",
        "subject",
        "task",
        "system",
        "age",
        "sex",
    ),
) -> SessionDataset:
    """
    Builds a :class:`SessionDataset` from a sessions CSV.

    Parameters
    ----------
    sessions_csv : str
        Path to a CSV indexing h5 sessions. Must contain a ``session`` column
        and columns named in ``info_cols``.
    h5_dir : str
        Directory containing ``{session}.h5`` files.
    window_len : int
        Number of samples per window.
    sfreq : float
        Sampling frequency in Hz. Defaults to 250 Hz.
    standardize : bool
        Apply per-session, per-channel z-score standardisation.
    include_sessions : Optional[Sequence[str]]
        If given, restrict to these session IDs.
    info_cols : Sequence[str]
        Columns copied from the CSV into each item's ``info`` dict.
    """
    df = pd.read_csv(sessions_csv)
    if include_sessions is not None:
        df = df[df["session"].isin(set(include_sessions))]
    if df.empty:
        raise ValueError("No sessions selected from CSV.")

    h5_dir_path = Path(h5_dir)
    sessions: List[H5Session] = []
    for _, row in df.iterrows():
        info = {c: row[c] for c in info_cols if c in row}
        h5_path = h5_dir_path / f"{row['session']}.h5"
        sessions.append(
            H5Session(
                h5_path=str(h5_path),
                window_len=window_len,
                sfreq=sfreq,
                info=info,
                standardize=standardize,
            )
        )
    return SessionDataset(sessions)


def load_session_array(path: str, picks: str = "misc") -> np.ndarray:
    """
    Load a parcellated FIF as a ``(n_samples, n_channels)`` float32 array.

    Bad-annotated segments are dropped (``reject_by_annotation="omit"``) so the
    returned array holds only good samples — matching the covariance/tokenisation
    convention used across the pipeline.

    Parameters
    ----------
    path : str
        Path to a parcellated ``*-raw.fif`` file.
    picks : str
        MNE picks selecting the parcel channels (parcels are stored as ``misc``).
    """
    import mne  # imported lazily so h5-only users need no MNE install

    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    data = raw.get_data(picks=picks, reject_by_annotation="omit", verbose="ERROR")
    return np.ascontiguousarray(data.T, dtype=np.float32)


class FIFSession(WindowedSession):
    """
    FIF-backed windowed session.

    Reads parcel time courses from a parcellated FIF via MNE-Python (``picks``,
    bad segments omitted).

    FIFs cannot be memory-mapped like h5, so the whole session array is loaded
    once per worker process and cached — a worker's peak memory scales with the
    number of sessions it touches.

    Parameters
    ----------
    fif_path : str
        Path to a parcellated ``*-raw.fif`` file.
    window_len, sfreq, info, standardize
        See :class:`WindowedSession`.
    picks : str
        Selected parcel type or channels.
    """

    def __init__(
        self,
        fif_path: str,
        window_len: int,
        sfreq: float,
        info: Dict[str, Any],
        standardize: bool = True,
        picks: str = "misc",
    ):
        self.fif_path = str(fif_path)
        self.picks = str(picks)
        super().__init__(window_len, sfreq, info, standardize)

    def _load_array(self) -> np.ndarray:
        return load_session_array(self.fif_path, picks=self.picks)

    # The whole array is the per-process resource; windows are in-memory slices.
    def _open(self) -> np.ndarray:
        return load_session_array(self.fif_path, picks=self.picks)

    def _read_window(self, resource: np.ndarray, start: int, end: int) -> np.ndarray:
        return resource[start:end, :]


def build_fif_dataset(
    sessions_csv: str,
    window_len: int,
    sfreq: float = 250.0,
    standardize: bool = True,
    include_sessions: Optional[Sequence[str]] = None,
    fif_col: str = "parc_file",
    picks: str = "misc",
    info_cols: Sequence[str] = (
        "session",
        "dataset",
        "subject",
        "task",
        "system",
        "age",
        "sex",
    ),
) -> SessionDataset:
    """
    Builds a :class:`SessionDataset` of :class:`FIFSession` from a sessions CSV.

    Each session's parcel data is read from the FIF path in ``row[fif_col]``
    (default ``"parc_file"``).

    Parameters
    ----------
    sessions_csv : str
        Path to a CSV with a ``session`` column, a ``fif_col`` column of FIF
        paths, and the columns named in ``info_cols``.
    window_len : int
        Number of samples per window.
    sfreq : float
        Sampling frequency in Hz.
    standardize : bool
        Apply per-session, per-channel z-score standardisation.
    include_sessions : Optional[Sequence[str]]
        If given, restrict to these session IDs.
    fif_col : str
        CSV column holding each session's FIF path.
    picks : str
        MNE picks selecting the parcel channels.
    info_cols : Sequence[str]
        Columns copied from the CSV into each item's ``info`` dict.
    """
    df = pd.read_csv(sessions_csv)
    if include_sessions is not None:
        df = df[df["session"].isin(set(include_sessions))]
    if df.empty:
        raise ValueError("No sessions selected from CSV.")
    if fif_col not in df.columns:
        raise ValueError(
            f"sessions CSV '{sessions_csv}' has no '{fif_col}' column of FIF paths."
        )

    sessions: List[FIFSession] = []
    for _, row in df.iterrows():
        info = {c: row[c] for c in info_cols if c in row}
        sessions.append(
            FIFSession(
                fif_path=str(row[fif_col]),
                window_len=window_len,
                sfreq=sfreq,
                info=info,
                standardize=standardize,
                picks=picks,
            )
        )
    return SessionDataset(sessions)
