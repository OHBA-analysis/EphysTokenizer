"""Example script for training baseline non-learnable tokenizers."""

# Import packages
import hydra
import logging
import mne
from glob import glob
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from ephys_tokenizer.configs import get_config
from ephys_tokenizer.models import MuTransformTokenizer, StandardQuantileTokenizer
from ephys_tokenizer.utils import plotting


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODELS = {
    "mu_transform_tokenizer": MuTransformTokenizer,
    "standard_quantile_tokenizer": StandardQuantileTokenizer
}


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    # ---------- Set Up ----------
    _logger.info("\n===== Configuration =====:\n" + OmegaConf.to_yaml(cfg))

    # Set main config
    run_dir = cfg.main.run_dir
    load = cfg.main.load

    # Load tokenizer model config
    model_config = OmegaConf.to_container(cfg.model_config, resolve=True)
    model_config = get_config(model_config)  # Config object
    model_cfg = model_config.config_class  # tokenizer-specific Config object
    model = MODELS[model_cfg.name]

    # Create directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "figures").mkdir(exist_ok=True)

    # ---------- Dataset ----------

    # Get data files
    data_dir = "/well/win-camcan/shared/spring23/src"
    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))[:50]  # use subset for example
    subject_ids = [Path(f).parent.name for f in data_files]

    # Prepare dataset and data module
    camcan_data = []
    for id in subject_ids:
        raw = mne.io.read_raw_fif(f"{data_dir}/{id}/sflip_parc-raw.fif", preload=True, verbose=False)
        data = raw.get_data(picks="misc", reject_by_annotation="omit", verbose=False).T
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        camcan_data.append(data)

    # ---------- Model Fitting ----------

    if not load:
        # Build and run tokenizer
        tokenizer = model(model_config)
        tokenizer.fit(camcan_data, clip=4)  # clip to 4 standard deviations
        tokenizer.save(run_dir)
        _logger.info(f"Fitting finished. Model saved to: {run_dir}")
    else:
        # Load model
        tokenizer = model.load_model(run_dir)

    # ---------- Visualization ----------

    # Plot subject-level PVEs
    pve = tokenizer.get_pve(camcan_data, n_jobs=8)
    print(f"Percentage of Variance Explained (PVE) - Average: {pve.mean()}")
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
    main()
