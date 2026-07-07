"""Example: tokenise and detokenise a continuous recording with EphysTokenizer.

This is the natural follow-on to ``train_etkn.py``. Given a trained model (as
saved by ``EphysTokenizerModule.save``), it:

1. Tokenises a continuous ``(n_samples, n_channels)`` signal into an integer token
   stream with :meth:`EphysTokenizerModule.tokenize_session` — overlap-and-stitch:
   it slides length-``L`` windows with stride ``L - 2M`` and keeps only each
   window's clean middle ``[M : L-M]``, so every token has full decoder context.
2. Reconstructs the signal from those tokens with
   :meth:`EphysTokenizerModule.reconstruct_session`.

A synthetic signal is used so the script is self-contained.

Usage
-----
    # 1. train + save a model:
    python train_etkn.py
    # 2. tokenise + detokenise with it:
    python tokenize_etkn.py --model-dir <run_dir_from_step_1>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule


def synthetic_signal(n_samples, n_channels, sfreq=250.0, seed=0):
    """A smooth multi-channel signal (sum of random oscillations) for the demo."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    x = np.zeros((n_samples, n_channels))
    for c in range(n_channels):
        for _ in range(3):
            freq = rng.uniform(2.0, 40.0)          # Hz
            phase = rng.uniform(0.0, 2 * np.pi)
            x[:, c] += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * freq * t + phase)
    x += 0.1 * rng.standard_normal((n_samples, n_channels))
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="directory with a trained model (model_state.pt, vocab.pkl, "
                         "config), as written by EphysTokenizerModule.save / train_etkn.py")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="length of the synthetic signal to tokenise")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--margin", type=int, default=0,
                    help="context margin M dropped at each window edge (0 keeps all "
                         "samples; >0, e.g. token_dim, gives full clean context)")
    ap.add_argument("--out", default="tokenize_example.png",
                    help="where to save the original-vs-reconstructed plot")
    args = ap.parse_args()

    # ---------- load the trained tokenizer ----------
    module = EphysTokenizerModule.load_model(args.model_dir)
    n_channels = module.config.n_channels

    # ---------- a continuous (n_samples, n_channels) signal to tokenise ----------
    n_samples = int(args.seconds * args.sfreq)
    signal = synthetic_signal(n_samples, n_channels, sfreq=args.sfreq)
    print(f"Signal:  {signal.shape}  "
          f"({args.seconds:g}s @ {args.sfreq:g}Hz, {n_channels} ch)")

    # ---------- tokenise ----------
    tokens = module.tokenize_session(signal, margin=args.margin,
                                     standardize=True, remap=True)
    print(f"Tokens:  {tokens.shape}  dtype={tokens.dtype}  "
          f"{len(np.unique(tokens))} distinct ids")

    # ---------- detokenise ----------
    recon = module.reconstruct_session(tokens)
    print(f"Recon:   {recon.shape}")

    # tokenize_session drops the first/last `margin` samples; reconstruct_session
    # crops to a whole number of length-L windows. Align the original accordingly.
    std = signal.std(axis=0)
    signal_std = (signal - signal.mean(axis=0)) / np.where(std == 0.0, 1.0, std)
    original = signal_std[args.margin:args.margin + recon.shape[0]]

    pve = 1.0 - np.var(original - recon) / np.var(original)
    print(f"PVE (synthetic, illustrative only): {pve:.3f}")

    # ---------- plot original vs reconstructed for a few channels ----------
    n_plot = min(3, n_channels)
    n_show = min(int(5 * args.sfreq), recon.shape[0])
    t = np.arange(n_show) / args.sfreq
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.0 * n_plot), sharex=True)
    axes = np.atleast_1d(axes)
    for c in range(n_plot):
        axes[c].plot(t, original[:n_show, c], label="original", lw=1.0)
        axes[c].plot(t, recon[:n_show, c], label="reconstructed", lw=1.0, alpha=0.8)
        axes[c].set_ylabel(f"ch {c}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("EphysTokenizer: tokenise_session → reconstruct_session")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved plot to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
