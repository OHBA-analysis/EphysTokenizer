"""Example script for training baseline non-learnable tokenizers."""

# Import packages
import argparse
import logging
import mne
import numpy as np
from glob import glob
from pathlib import Path
from typing import Optional

from ephys_tokenizer.configs import get_config
from ephys_tokenizer.models import MuTransformTokenizer, StandardQuantileTokenizer
from ephys_tokenizer.utils import plotting


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


models = {
    "mu_transform_tokenizer": MuTransformTokenizer,
    "standard_quantile_tokenizer": StandardQuantileTokenizer
}


def main(
    config_path: str,
    run_dir: str,
    load: Optional[str] = False,
):
    # Load config
    config = get_config(config_path)
    cfg = config.config_class
    model = models[cfg.name]

    # Get directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "figures").mkdir(exist_ok=True)

    data_dir = "/well/win-camcan/shared/spring23/src"

    # Select subset of the data
    train_idx = np.array([
        38, 57, 421, 534, 413, 146, 245, 152, 410, 139, 79, 583, 489,
        67, 218, 260, 342, 118, 372, 51, 592, 289, 598, 504, 538, 171,
       320, 137, 41, 157, 341, 596, 375, 502, 32, 590, 560, 37, 155,
       495, 142, 183, 332, 339, 353, 518, 194, 475, 93, 64,
    ])  # selected using the numpy random generator with seed=813
    print(f"Number of training subjects: {len(train_idx)}")

    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
    data_files = [data_files[i] for i in train_idx]
    subject_ids = sorted([Path(f).parent.name for f in data_files])
    # NOTE: It is important to sort the subject IDs, as they get sorted automatically
    #       inside the CamcanGlasser dataset.
    #       If not, this creates mismatch between subjects in `plot_fitted_signal()`.

    # ---------- Dataset ----------

    # Prepare dataset and data module
    camcan_data = []
    for id in subject_ids:
        raw = mne.io.read_raw_fif(f"{data_dir}/{id}/sflip_parc-raw.fif", preload=True, verbose=False)
        data = raw.get_data(picks="misc", reject_by_annotation="omit", verbose=False).T
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        camcan_data.append(data)

    # ---------- Model Training ----------

    if not load:
        # Build and fit tokenizer
        tokenizer = model(config)
        tokenizer.fit(camcan_data, clip=4)  # clip to 4 standard deviations
        tokenizer.save(run_dir)
        _logger.info(f"Training finished. Model saved to: {run_dir}")
    else:
        # Load model
        tokenizer = model.load_model(run_dir)

    # ---------- Visualization ----------

    # Compute PVE
    pve = tokenizer.get_pve(camcan_data, n_jobs=8)
    print(f"Percentage of Variance Explained (PVE): {pve}")
    plotting.plot_pve(pve, plot_dir=f"{run_dir}/figures")

    # Plot signals reconstructed from tokenized data (for one session)
    tokens = tokenizer.tokenize_data(
        camcan_data, concatenate=False, n_jobs=8
    )
    reconstructed_data = tokenizer.reconstruct_data(
        tokens, concatenate=False, n_jobs=8
    )
    plotting.plot_fitted_signal(
        original_data_path=f"{data_dir}/{subject_ids[0]}/sflip_parc-raw.fif",
        reconstructed_data=reconstructed_data,
        subject_idx=0,
        plot_dir=f"{run_dir}/figures",
    )

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Non-Learnable Baselines")
    p.add_argument("--config", type=str, required=True, help="Path to config.yml.")
    p.add_argument("--run_dir", type=str, required=True, help="Directory to save the trained model.")
    p.add_argument("--load", action="store_true", help="Whether to load a trained model.")
    args = p.parse_args()

    main(
        config_path=args.config,
        run_dir=args.run_dir,
        load=args.load,
    )
