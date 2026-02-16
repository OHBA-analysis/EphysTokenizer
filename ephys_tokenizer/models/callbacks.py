"""Callback functions for the tokenizer training."""

# Import packages
import logging
import numpy as np
import os
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional


_logger = logging.getLogger(__name__)


class CheckpointCallback(pl.Callback):
    """
    Callback to save model checkpoints during training.

    Parameters
    ----------
    save_freq : int
        Frequency (in epochs) to save checkpoints.
    checkpoint_dir : str
        Directory to save checkpoints.
    """
    def __init__(self, save_freq: int, checkpoint_dir: str):
        super().__init__()
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_path = str(Path(self.checkpoint_dir).parent / "history.pkl")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.save_freq == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt-epoch{epoch}.ckpt")
            _logger.info(f"\nSaving checkpoint to {checkpoint_path}.")
            trainer.save_checkpoint(checkpoint_path, weights_only=False)


class TemperatureAnnealingCallback(pl.Callback):
    """
    Callback to linearly anneal the temperature of the token weights.

    Note: If `end_temperature` is set exactly to 0, the token weights will be hard
          one-hot encoded, and gradients will not flow through the layer.

    Parameters
    ----------
    n_stages : int
        Number of stages for temperature annealing.
    n_epochs : int
        Total number of annealing epochs.
    start_temperature : float, optional
        Initial temperature for annealing.
    end_temperature : float, optional
        Final temperature for annealing.
    multi_gpu : bool, optional
        Whether to use multi-GPU training.
    """
    def __init__(
        self,
        n_stages: int,
        n_epochs: int,
        start_temperature: Optional[float] = 1.0,
        end_temperature: Optional[float] = 1e-3,
        multi_gpu: Optional[bool] = False,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.n_epochs = n_epochs
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.multi_gpu = multi_gpu

        self.temperatures = np.linspace(start_temperature, end_temperature, n_stages)
        self.n_epochs_per_stage = max(1, n_epochs // n_stages)
        # NOTE: last stage may be longer if n_epochs is not divisible

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        stage = min(self.n_stages - 1, epoch // self.n_epochs_per_stage)
        
        temperature = float(self.temperatures[stage])
        token_weights_layer = pl_module.token_weights_layer
        token_weights_layer.temperature = temperature
        
        pl_module.log("train/temperature", temperature, on_epoch=True, prog_bar=True, sync_dist=self.multi_gpu)
        _logger.info(f"\nSet temperature to {temperature} for epoch {epoch}.")

    def on_validation_end(self, trainer, pl_module):
        # Log current temperature at the end of validation
        token_weights_layer = pl_module.token_weights_layer
        temperature = float(token_weights_layer.temperature)
        _logger.info(f"\nSet temperature to {temperature} for validation.")
