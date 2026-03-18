"""Example script for training EphysTokenizer."""

# Import packages
import hydra
import logging
from glob import glob
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from ephys_tokenizer.configs import get_config
from ephys_tokenizer.data.dataloader import CamcanGlasserDataModule
from ephys_tokenizer.models import callbacks
from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule
from ephys_tokenizer.utils import plotting
from ephys_tokenizer.utils.train import get_history
from pnpl.datasets import CamcanGlasser


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="ex_etkn", config_name="config")
def main(cfg: DictConfig):
    # ---------- Set Up ----------
    _logger.info("\n===== Configuration =====:\n" + OmegaConf.to_yaml(cfg))

    # Set main config
    run_dir = cfg.main.run_dir
    gpus = cfg.main.gpus
    precision = cfg.main.precision
    deterministic = cfg.main.deterministic
    seed = cfg.main.seed
    checkpoint = cfg.main.checkpoint

    # Load tokenizer model config
    model_config = get_config(cfg.model_config)  # Config object
    model_cfg = model_config.config_class  # tokenizer-specific Config object

    # Set model training config
    batch_size = model_cfg.training.batch_size
    n_epochs = model_cfg.training.n_epochs
    multi_gpu = model_cfg.training.multi_gpu

    # Get directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "checkpoints").mkdir(exist_ok=True)
    (Path(run_dir) / "figures").mkdir(exist_ok=True)

    # Set seed (for reproducibility)
    pl.seed_everything(seed, workers=True)

    # ---------- Dataset ----------

    # Get data files
    data_dir = "/well/win-camcan/shared/spring23/src"
    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))[:50]  # use subset for example
    subject_ids = [Path(f).parent.name for f in data_files]

    # Prepare dataset and data module
    camcan_data = CamcanGlasser(
        data_path=data_dir,
        window_len=200,
        info=["subject", "dataset", "subject_id", "session"],
        picks="misc",
        reject_by_annotation="omit",
        standardize=True,
        include_subjects=subject_ids,
        verbose=False,
    )
    camcan_datamodule = CamcanGlasserDataModule(
        dataset=camcan_data,
        batch_size=batch_size,
        val_split=0,
        split_method="subject_window",
        is_distributed=multi_gpu,
        seed=seed,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # ---------- Model Training ----------

    if checkpoint is None:
        # Build network via Lightning module
        pl_module = EphysTokenizerModule(model_config)

        # Set logger
        logger = CSVLogger(save_dir=run_dir, name="csv_logs")
        log_dir = Path(run_dir) / "csv_logs/version_0"

        # Set callbacks
        checkpoint_callback = callbacks.CheckpointCallback(
            save_freq=1, checkpoint_dir=f"{run_dir}/checkpoints"
        )
        temperature_callback = callbacks.TemperatureAnnealingCallback(
            n_stages=model_cfg.callback.temperature_annealing["n_stages"],
            n_epochs=model_cfg.callback.temperature_annealing["n_annealing_epochs"],
            multi_gpu=multi_gpu,
        )
        cbs = [checkpoint_callback, temperature_callback]

        # Set trainer
        trainer_kwargs = dict(
            max_epochs=int(n_epochs),
            logger=logger,
            callbacks=cbs,
            deterministic=deterministic,
            precision=int(precision),
        )
        if gpus and gpus > 0:
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = gpus

        trainer = pl.Trainer(**trainer_kwargs)

        # Run training via the module wrapper (refactors vocab after training)
        pl_module.fit(trainer=trainer, datamodule=camcan_datamodule)
        pl_module.save(run_dir)  # save model weights and token vocab
        get_history(log_dir, save_dir=run_dir)  # save training history
        _logger.info(f"Training finished. Model saved to: {run_dir}")

    else:
        # Load model
        pl_module = EphysTokenizerModule.load_model(run_dir, checkpoint=checkpoint)

        # Set up data module for testing
        camcan_datamodule.setup(stage="test")

    # ---------- Visualization ----------
    # NOTE: Using `full_dataloader()` indicates we are using the full data subset defined
    #       above, not the entire Cam-CAN dataset.

    # Compute PVE
    pve = pl_module.get_pve(dataloader=camcan_datamodule.full_dataloader())
    print(f"Percentage of Variance Explained (PVE) - Average: {pve.mean()}")
    plotting.plot_pve(pve, plot_dir=f"{run_dir}/figures")

    # Plot token kernel response
    token_response, input = pl_module.get_token_kernel_response(
        dataloader=camcan_datamodule.full_dataloader(),
        input="impulse",
    )
    plotting.plot_token_response(token_response, input, plot_dir=f"{run_dir}/figures")

    # Plot token counts histogram
    plotting.plot_token_counts(
        vocab=f"{run_dir}/vocab.pkl", plot_dir=f"{run_dir}/figures"
    )

    # Plot signals reconstructed from tokenized data (for one session)
    tokens, token_weights = pl_module.tokenize_data(
        camcan_datamodule.full_dataloader(),
        batch_size=32,
        remap=False,
        return_weights=True,
        num_workers=8,
    )
    reconstructed_data = pl_module._reconstruct_data(tokens)
    plotting.plot_fitted_signal(
        original_data_path=f"{data_dir}/{subject_ids[0]}/sflip_parc-raw.fif",
        reconstructed_data=reconstructed_data,
        token_weights=token_weights,
        subject_idx=0,
        plot_dir=f"{run_dir}/figures",
    )

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
