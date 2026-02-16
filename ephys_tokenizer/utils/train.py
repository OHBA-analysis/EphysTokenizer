"""Utility functions for model training."""

# Import packages
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, Optional


def get_history(
    log_dir: str, save_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Loads training history from a CSV log file.

    Parameters
    ----------
    log_dir : str
        Directory containing the log files.
    save_dir : str, optional
        Directory to save the training history pickle file.
        If None, the history will not be saved.

    Returns
    -------
    history : Dict[str, np.ndarray]
        Dictionary containing the training history.
    """
    # Read metric log file
    log_path = Path(log_dir) / "metrics.csv"
    if log_path.exists():
        df = pd.read_csv(log_path)
    else:
        raise FileNotFoundError(
            f"No metrics.csv found in {log_dir}. Cannot load training history."
        )
    
    # Collect training history
    history = {}
    history["loss"] = df["train/loss"].to_numpy()
    history["temperature"] = df["train/temperature"].to_numpy()
    
    # Save history
    if save_dir:
        save_path = Path(save_dir) / "history.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(history, f)
    
    return history


def unwrap_dataset(dl: DataLoader) -> ConcatDataset:
    """
    Recursively unwraps `.dataset` attributes from the PyTorch dataloader
    until the base dataset is reached.

    Parameters
    ----------
    dl : DataLoader
        A PyTorch DataLoader to unwrap.

    Returns
    -------
    dl : ConcatDataset
        The base dataset.
    """
    while hasattr(dl, "dataset"):
        dl = dl.dataset
    return dl
