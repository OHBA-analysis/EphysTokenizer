"""Utility functions for visualizing post-hoc analysis results."""

# Import packages
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union


def _rough_square_axes(n_plots):
    """
    Gets the appropriate square axis layout for a given number of plots.

    Given `n_plots`, find the side lengths of the rectangle which gives
    the closest layout to a square grid of axes.

    Parameters
    ----------
    n_plots : int
        Number of plots to arrange.

    Returns
    -------
    short : int
        Number of axes on the short side.
    long : int
        Number of axes on the long side.
    empty : int
        Number of axes left blank from the rectangle.
    """
    long = np.floor(n_plots**0.5).astype(int)
    short = np.ceil(n_plots**0.5).astype(int)
    if short * long < n_plots:
        short += 1
    empty = short * long - n_plots
    return short, long, empty


def plot_pve(
    pve: np.ndarray,
    plot_dir: Optional[str] = None,
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the percentage of variance explained (PVE).

    Parameters
    ----------
    pve : np.ndarray
        Percentage of variance explained.
    plot_dir : str, optional
        Directory to save the plot.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : plt.Axes
        Axes object.
    """
    # Plot a histogram of PVEs
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.hist(pve, bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("PVE (%)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Percentage of Variance Explained (Avg: {:.2f}%)".format(pve.mean()))
    plt.tight_layout()
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f"{plot_dir}/pve_histogram.png")
        plt.close(fig)
    else:
        return fig, ax


def plot_token_response(
    token_response: np.ndarray,
    input: np.ndarray,
    plot_dir: Optional[str] = None,
) -> None:
    """
    Plots a stimulus response of each token kernel.

    Parameters
    ----------
    token_response : np.ndarray
        Response of each token kernel.
    input : np.ndarray
        Stimulus input to get kernel response for. 
    plot_dir : str, optional
        Directory to save the plot.
    """
    # Number of tokens
    n_tokens = len(token_response)

    # Limit number of tokens to plot
    if n_tokens > 30:
        n_tokens = 30
        token_response = token_response[:n_tokens]  # select top 30 tokens

    # Plot stimulus responses for each token
    short, long, _ = _rough_square_axes(n_tokens)
    fig, axes = plt.subplots(nrows=short, ncols=long, figsize=(2 * short, 3 * long))
    axes = axes.flatten()
    for n, resp in enumerate(token_response):
        axes[n].plot(resp, label="Token Response" if n == 0 else "")
        axes[n].plot(input, "r", label="Input" if n == 0 else "")
        axes[n].set_ylim([-1.1, 1.1])
    for ax in axes[n_tokens:]:
        ax.axis("off")
    fig.legend()
    plt.tight_layout()

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f"{plot_dir}/token_response.png")
        plt.close(fig)


def plot_token_counts(
    vocab: Union[dict, str],
    plot_dir: Optional[str] = None,
) -> None:
    """
    Plots a histogram of token counts over all subjects/sessions.

    Parameters
    ----------
    vocab : Union[dict, str]
        Vocabulary data for plotting. Should be either a dictionary containing
        vocabulary or a path to a vocabulary file.
    plot_dir : str, optional
        Directory to save the plot.
    """
    # Get vocabulary
    if isinstance(vocab, str):
        with open(vocab, "rb") as f:
            vocab = pickle.load(f)

    total_token_counts = vocab["total_token_counts"]

    # Plot a histogram of token counts
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    axes.bar(
        range(1, total_token_counts.shape[0] + 1),
        total_token_counts,
        color="skyblue",
        edgecolor="black",
    )
    axes.set_xlabel("Token Index")
    axes.set_ylabel("Number of Occurrences")
    axes.set_title(f"Token Histogram (N={len(total_token_counts)})")
    plt.tight_layout()

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f"{plot_dir}/token_counts.png")
        plt.close(fig)


def plot_fitted_signal(
    original_data_path: str,
    reconstructed_data: Union[np.ndarray, List[np.ndarray]],
    token_weights: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    subject_idx: Optional[int] = 0,
    plot_dir: Optional[str] = None,
) -> None:
    """
    Plots a signal reconstructed from tokenized data and its token weights.

    Parameters
    ----------
    original_data_path : str
        Path to the original data file.
    reconstructed_data : Union[np.ndarray, List[np.ndarray]]
        Reconstructed data from the tokenized input.
    token_weights : Union[np.ndarray, List[np.ndarray]], optional
        Token weights for the reconstructed data.
    subject_idx : int, optional
        Index of the subject to plot.
    plot_dir : str, optional
        Directory to save the plot.
    """
    # Read original data
    file_extn = Path(original_data_path).suffix  # file extension
    if file_extn == ".fif":
        raw = mne.io.read_raw_fif(original_data_path, preload=True, verbose=False)
        data = raw.get_data(
            picks="misc",
            reject_by_annotation="omit",
            verbose=False,
        )
    elif file_extn == ".npy":
        data = np.load(original_data_path)

    # Get correct data shape
    if data.shape[0] < data.shape[1]:  # assumes n_samples > n_channels
        data = data.T
    
    # Standardize original data
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    original_data = (data - mean) / std

    # Get reconstructed data and token weights
    reconstructed_data = reconstructed_data[subject_idx]
    if token_weights is not None:
        token_weights = token_weights[subject_idx]

    # Match the data lengths
    min_length = reconstructed_data.shape[0]
    original_data = original_data[:min_length]

    # Plot data signals and token weights
    n_channels = min(original_data.shape[1], 3)  # number of channels to plot
    start_idx, end_idx = 200, 500  # start and end indices to plot
    x = np.arange(start_idx, end_idx)
    for n in range(n_channels):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 5))
        axes[0].plot(x, original_data[start_idx:end_idx, n], label="Original")
        axes[0].plot(x, reconstructed_data[start_idx:end_idx, n], label="Fitted")
        axes[0].set_title(f"Channel {n}: Data Signals")
        axes[0].legend()
        if token_weights is not None:
            axes[1].plot(x, token_weights[start_idx:end_idx, n, :])
            axes[1].set_title(f"Token Weights")
        plt.tight_layout()

        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(f"{plot_dir}/fitted_signal_ch{n}.png")
            plt.close(fig)
