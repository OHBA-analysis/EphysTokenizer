"""PyTorch DataLoader and Lightning DataModule for tokenizer training."""

# Import packages
from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import random
import torch
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


def _collate_camcan(batch: Sequence[dict]) -> Dict[str, Any]:
    """
    Collate function for CamcanGlasser items.

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
    collate_fn=_collate_camcan,
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


class CamcanGlasserDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the CamCAN dataset.

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
        the same order as subjects you provided to CamcanGlasser.

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
