"""
Implementation of the EphysTokenizer model.

Mathematical Notation:
  - B: batch size
  - L: sequence length
  - C: number of channels
  - T: number of time samples
  - N: number of subjects/sessions
  - N_t: number of tokens
"""

# Import packages
import glob
import logging
import os
import pickle
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from ephys_tokenizer.configs import Config, get_config
from ephys_tokenizer.models.layers import (
    TokenWeightsLayer,
    MSELossLayer,
    EncoderLayer,
    DecoderLayer,
)
from ephys_tokenizer.utils.initializer import init_model_weights
from ephys_tokenizer.utils.train import unwrap_dataset


_logger = logging.getLogger(__name__)


def _resolve_optimizer(
        params: Iterable, optimizer_descriptor: Any
    ) -> torch.optim.Optimizer:
    """
    Resolves a PyTorch optimizer from an optimizer descriptor.

    Parameters
    ----------
    params : Iterable
        The parameters to optimize.
    optimizer_descriptor : Any
        The optimizer descriptor. This can be:
            - a callable: optimizer_factory(params) -> optimizer instance
            - a torch optimizer instance -> returned directly
            - a tuple/list: (torch.optim.OptimizerClass, {"lr":..., ...})
            - a TF optimizer object or dict from get_config() -> attempt to map to torch optimizer
    """
    if callable(optimizer_descriptor):
        return optimizer_descriptor(params)

    if isinstance(optimizer_descriptor, torch.optim.Optimizer):
        return optimizer_descriptor

    if isinstance(optimizer_descriptor, (list, tuple)) and len(optimizer_descriptor) >= 1:
        optim_cls = optimizer_descriptor[0]
        optim_kwargs = optimizer_descriptor[1] if len(optimizer_descriptor) > 1 else {}
        return optim_cls(params, **optim_kwargs)

    if isinstance(optimizer_descriptor, dict):
        name = optimizer_descriptor.get("name", "adam").lower()
        lr = optimizer_descriptor.get("learning_rate", 1e-3)
        eps = optimizer_descriptor.get("eps", 1e-7)
        if "adam" in name:
            return optim.Adam(params, lr=lr, eps=eps)
        if "sgd" in name:
            return optim.SGD(params, lr=lr)
        return optim.Adam(params, lr=lr, eps=eps)

    raise ValueError("Unsupported optimizer descriptor.")


class EphysTokenizer(nn.Module):
    """
    EphysTokenizer class.

    Parameters
    ----------
    config : Config
        Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config.config_class

        # Build the model components
        self.encoder_layer = EncoderLayer(
            rnn_type=self.config.rnn_type,
            rnn_n_layers=self.config.rnn_n_layers,
            rnn_n_units=self.config.rnn_n_units,
        )
        self.token_weights_layer = TokenWeightsLayer(
            input_dim=self.config.rnn_n_units,
            output_dim=self.config.n_tokens,
        )
        self.decoder_layer = DecoderLayer(
            n_channels=self.config.n_channels,
            sequence_length=self.config.sequence_length,
            n_tokens=self.config.n_tokens,
            token_dim=self.config.token_dim,
            token_kernel_padding=self.config.token_kernel_padding,
        )
        self.mse_loss_layer = MSELossLayer()

        # Initialize model weights
        init_model_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the encoder
        encoder_output = self.encoder_layer(x)
        # x.shape: (B, L, C)
        # encoder_output.shape: (B * C, L, rnn_n_units)

        # Compute token weights
        token_weights = self.token_weights_layer(encoder_output)
        # token_weights.shape: (B * C, L, N_t)

        # Decode the token weights
        reconstructed_data, token_weights = self.decoder_layer(token_weights)
        # reconstructed_data.shape: (B, L, C)
        # token_weights.shape: (B, L, C, N_t)

        # Compute the loss
        loss = self.mse_loss_layer(x, reconstructed_data)

        return loss, reconstructed_data, token_weights


class EphysTokenizerModule(pl.LightningModule):
    """
    EphysTokenizer Lightning Module.

    Parameters
    ----------
    config : Config
        Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.base_config = config
        self.config = config.config_class
        self.model = EphysTokenizer(config)
        self.vocab: Dict[str, Any] = {}

        # Expose network submodules for convenience
        self.encoder_layer = self.model.encoder_layer
        self.token_weights_layer = self.model.token_weights_layer
        self.decoder_layer = self.model.decoder_layer

        # Save meta-data hyperparameters
        try:
            self.save_hyperparameters(
                {"model_name": self.config.name},
                ignore=["config", "model"],
            )
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        loss, _, _ = self.forward(batch["data"].float())
        self.log(
            "train/loss", loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=self.config.batch_size,
            sync_dist=self.config.multi_gpu,
        )
        # NOTE: on_epoch logs the mean across all steps (batches) in the epoch.
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        loss, _, _ = self.forward(batch["data"].float())
        self.log(
            "val/loss", loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=self.config.batch_size,
            sync_dist=self.config.multi_gpu,
        )
        # NOTE: on_epoch logs the mean across all steps (batches) in the epoch.
        return loss

    def configure_optimizers(self):
        """
        Configures optimizers for training.
        """
        # Validation
        if self.config is None or not hasattr(self.config, "optimizer") or not self.config.optimizer:
            raise ValueError("Optimizer is not defined in the training configuration.")

        # Get optimizer
        optim_description = self.config.optimizer
        optimizer = _resolve_optimizer(self.parameters(), optim_description)
        return optimizer

    def fit(
        self,
        trainer: pl.Trainer,
        datamodule: pl.LightningDataModule,
        **kwargs,
    ):
        """
        Fits the model using the specified trainer and datamodule.
        """
        # Run training
        trainer.fit(self, datamodule=datamodule, weights_only=False, **kwargs)

        # Refactor vocabularies
        train_dl = datamodule.train_dataloader()
        self.refactor_vocab(train_dl)

    # -----------------------------
    # Tokenization & Reconstruction
    # -----------------------------

    def tokenize_data(
        self,
        dataloader: DataLoader,
        batch_size: Optional[int] = None,
        remap: Optional[bool] = True,
        concatenate: Optional[bool] = False,
        return_weights: Optional[bool] = False,
        device: Optional[str] = None,
        num_workers: Optional[int] = 4,
    ) -> Tuple[
            Union[np.ndarray, List[np.ndarray]],
            Union[np.ndarray, List[np.ndarray]],
        ]:
        """
        Tokenizes data using the trained model.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader to get data for tokenization.
        batch_size : int, optional
            Batch size to use for tokenization.
        remap : bool, optional
            Whether to remap tokens to their refactored labels.
        concatenate : bool, optional
            Whether to concatenate tokenized sequences.
        return_weights : bool, optional
            Whether to return token weights.
        device : str, optional
            The device to use for tokenization.
        num_workers : int, optional
            The number of worker processes to use for data loading.

        Returns
        -------
        tokenized_results : Tuple
            The tokenized data and (optionally) token weights.
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        _logger.info(f"Device for tokenization: {device}")

        # Set batch size
        batch_size = batch_size or self.config.batch_size

        # Unify torch dataset type
        dataset = unwrap_dataset(dataloader)
        if not isinstance(dataset, ConcatDataset):
            raise RuntimeError("Incorrect dataset type.")

        # Get start and end sequence indices for each subject
        ranges = []
        offset = 0
        for ds in dataset.datasets:
            length = len(ds)
            subj_id = getattr(ds, "subject", None)
            ranges.append((subj_id, offset, offset + length))
            offset += length
        n_total_sequences = len(dataset)

        # Build a DataLoader that returns batched tensors
        worker_dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
            collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None,
        )

        # Put model in evaluation mode
        model = self.model.to(device).eval()
        sequence_length = self.config.sequence_length
        n_channels = self.config.n_channels
        n_tokens = self.config.n_tokens

        # Preallocate arrays for tokens (and weights if requested)
        tokens = torch.empty(
            (n_total_sequences, sequence_length, n_channels),
            dtype=torch.int64,
            device=device,
        )
        weights = None
        if return_weights:
            weights = torch.empty(
                (n_total_sequences, sequence_length, n_channels, n_tokens),
                dtype=torch.float32,
                device="cpu",
            ).pin_memory()

        idx = 0
        with torch.inference_mode():
            for batch in tqdm(worker_dl, desc="Tokenizing batches ...", total=len(worker_dl)):
                x = batch["data"]
                if x.device != device:
                    x = x.to(device, non_blocking=True)

                _, _, tw = model(x)  # shape: (B, L, C, N_t)
                t = tw.argmax(dim=-1)  # shape: (B, L, C)
                bsz = t.shape[0]
                tokens[idx:idx + bsz].copy_(t)

                if return_weights:
                    weights[idx:idx + bsz].copy_(tw, non_blocking=True)

                idx += bsz

            # Synchronize CUDA to ensure all streams are finished
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Move all tensors to CPU at once
        tokens = tokens.cpu().numpy()
        if return_weights:
            weights = weights.numpy()

        # Split tokens (and weights) by subject ranges
        all_tokens = []
        all_weights = []
        for s, start, end in ranges:
            all_tokens.append(tokens[start:end].reshape(-1, n_channels))
            if return_weights:
                all_weights.append(weights[start:end].reshape(-1, n_channels, n_tokens))
        # all_tokens.shape: (N, T, C)
        # all_weights.shape: (N, T, C, N_t)

        # Remap tokens to refactored tokens
        if remap:
            if not self.vocab:
                raise ValueError("Vocabulary is empty. Call refactor_vocab() first.")
            all_tokens = [self.vocab["label_map"][t] for t in all_tokens]

        if concatenate:
            all_tokens = np.concatenate(all_tokens, axis=0)
            if return_weights:
                all_weights = np.concatenate(all_weights, axis=0)

        return (all_tokens, all_weights) if return_weights else all_tokens

    def refactor_vocab(
        self,
        dataloader: DataLoader,
        sort: Optional[bool] = True,
        trim: Optional[bool] = True,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Refactors the vocabulary based on the data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader for the time series data.
        sort : bool, optional
            Should we sort the tokens by frequency? Defaults to True.
        trim : bool, optional
            Should we remove tokens with zero frequency? Defaults to True.
        batch_size : int, optional
            The batch size to use for tokenization.
            Defaults to None, in which case the batch size in the training
            config will be used.
        """
        _logger.info("Refactoring vocabulary ...")

        # Get hyperparameters
        n_tokens = self.config.n_tokens
        batch_size = batch_size or self.config.batch_size

        # Tokenize data by subject
        tokens = self.tokenize_data(
            dataloader,
            batch_size=batch_size,
            remap=False,
            concatenate=False,
            return_weights=False,
            device=None,
        )  # shape: (N, T, C)

        # Count tokens across samples and channels for each subject
        token_counts = np.array(
            [
                np.bincount(t.flatten().astype(np.int64), minlength=n_tokens)
                for t in tokens
            ],
            dtype=np.int32,
        )

        # Get token order based on token counts
        if sort:
            token_order = np.argsort(np.sum(token_counts, axis=0))[::-1]
        else:
            token_order = np.arange(n_tokens, dtype=np.int32)

        # Remove all token indices with zero total counts
        if trim:
            nonzero_mask = np.sum(token_counts, axis=0)[token_order] > 0
            token_order = token_order[nonzero_mask]

        # Apply trimming and ordering to token counts
        token_counts = token_counts[:, token_order]

        # Get labels from token orders
        label_map = np.zeros(n_tokens, dtype=np.int32)
        label_map[token_order] = np.arange(len(token_order), dtype=np.int32) + 1

        self.vocab = {
            "token_order": token_order,
            "token_counts": token_counts,
            "total_token_counts": np.sum(token_counts, axis=0),
            "label_map": label_map,
        }

    def _reconstruct_data(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        concatenate: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reconstructs data from tokens.

        Parameters
        ----------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens to reconstruct.
            Shape of tokens of each subject/session: (n_samples, n_channels).
        concatenate : bool, optional
            Whether to concatenate the reconstructed data over all subjects/sessions.
            Defaults to False.
        device : torch.device, optional
            The device to perform the reconstruction on. Defaults to None.

        Returns
        -------
        reconstructed_data : Union[np.ndarray, List[np.ndarray]]
            The reconstructed data.
            Shape of reconstructed data of each subject/session: (n_samples, n_channels).
        """
        # Validation
        if device is None:
            device = next(self.model.parameters()).device
        device = torch.device(device)

        if not isinstance(tokens, list):
            tokens = [tokens]

        # Get hyperparameters
        n_tokens = self.config.n_tokens
        n_channels = self.config.n_channels
        sequence_length = self.config.sequence_length
        batch_size = self.config.batch_size

        # Get layer
        decoder_layer = self.model.decoder_layer
        token_basis_layer = self.model.decoder_layer.token_basis_layer

        decoder_layer = decoder_layer.to(device)
        token_basis_layer = token_basis_layer.to(device)

        # Helper function for reconstruction for each subject/session
        def _reconstruct_data_per_session(t):
            # Reshape token array by sequence length
            n_sequences = t.shape[0] // sequence_length
            t = t.reshape(n_sequences, sequence_length, n_channels)

            # Batch token array
            dataset = TensorDataset(torch.from_numpy(t).long())
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=4, pin_memory=True,
            )

            # Reconstruct for a single subject/session
            _reconstructed_data = []
            with torch.inference_mode():
                for batch in dataloader:
                    batch = batch[0].to(device)  # shape: (B, L, C)
                    
                    batch = batch.permute(0, 2, 1)
                    batch = batch.reshape(-1, sequence_length)
                    batch = nn.functional.one_hot(batch, n_tokens).float()
                    batch = batch.permute(0, 2, 1).to(device)
                    # shape: (B * C, N_t, L)

                    batch = decoder_layer._pad(batch)
                    batch = torch.sum(token_basis_layer(batch), dim=1)
                    # shape: (B * C, L)

                    batch = batch.reshape(-1, n_channels, sequence_length)
                    batch = batch.permute(0, 2, 1)
                    # shape: (B, L, C)

                    batch = batch.cpu().numpy()

                    _reconstructed_data.append(batch.reshape(-1, n_channels))

            return np.concatenate(_reconstructed_data, axis=0)

        reconstructed_data = []
        for t in tqdm(tokens, desc="Reconstructing data", total=len(tokens)):
            reconstructed_data.append(_reconstruct_data_per_session(t))

        if concatenate or len(reconstructed_data) == 1:
            reconstructed_data = np.concatenate(reconstructed_data)

        return reconstructed_data

    def reconstruct_data(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        concatenate: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reconstructs data from tokens using the refactored vocabularies.

        Parameters
        ----------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens to reconstruct.
            Shape of tokens per each subject/session: (n_samples, n_channels).
        concatenate : bool, optional
            Whether to concatenate the reconstructed data over all subjects/sessions.
            Defaults to False.
        device : torch.device, optional
            The device to perform the reconstruction on. Defaults to None.

        Returns
        -------
        reconstructed_data : Union[np.ndarray, List[np.ndarray]]
            The reconstructed data.
            Shape of reconstructed data per each subject/session: (n_samples, n_channels).
        """
        # Validation
        if not self.vocab:
            raise ValueError("Vocabulary is empty. Call refactor_vocab() first.")

        if not isinstance(tokens, list):
            tokens = [tokens]

        # Remap refactored tokens to original tokens
        unused_token_labels = np.where(self.vocab["label_map"] == 0)[0]
        if unused_token_labels.size == 0:
            raise ValueError("No unused token labels found.")

        remapped_tokens = []
        for t in tokens:
            remapped_t = (
                np.ones(t.shape, dtype=np.int32) * unused_token_labels[0]
            )  # use first of all unused tokens for outliers
            t -= 1
            remapped_t[t >= 0] = self.vocab["token_order"][t[t >= 0]]
            remapped_tokens.append(remapped_t)

        return self._reconstruct_data(remapped_tokens, concatenate, device)

    # -----------------
    # Post hoc analysis
    # -----------------

    def get_pve(
        self,
        dataloader: DataLoader,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        num_workers: Optional[int] = 4,
    ) -> np.ndarray:
        """
        Computes the percentage of variance explained by the tokens.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the input time series data.
        batch_size: int, optional
            Batch size to use for data loading. Defaults to None.
        num_workers : int, optional
            Number of workers to use for data loading. Defaults to 4.

        Returns
        -------
        pve : np.ndarray
            The percentage of variance explained by the tokens for each subject/session.
        """
        _logger.info("Getting percentage of variance explained ...")

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        _logger.info(f"Device for PVE computation: {device}")

        # Set batch size
        batch_size = batch_size or self.config.batch_size

        # Unify torch dataset type
        dataset = unwrap_dataset(dataloader)
        if not isinstance(dataset, ConcatDataset):
            raise RuntimeError("Incorrect dataset type.")
        
        # Get start and end sequence indices for each subject
        ranges = []
        offset = 0
        for ds in dataset.datasets:
            length = len(ds)
            subj_id = getattr(ds, "subject", None)
            ranges.append((subj_id, offset, offset + length))
            offset += length
        n_total_sequences = len(dataset)

        # Build a DataLoader that returns batched tensors
        worker_dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
            collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None,
        )

        # Put model in evaluation mode
        model = self.model.to(device).eval()
        sequence_length = self.config.sequence_length
        n_channels = self.config.n_channels

        # Preallocate arrays for tokens (and weights if requested)
        original_x = torch.empty((n_total_sequences, sequence_length, n_channels), dtype=torch.float32, device=device)
        reconstructed_x = torch.empty((n_total_sequences, sequence_length, n_channels), dtype=torch.float32, device=device)

        idx = 0
        with torch.inference_mode():
            for batch in tqdm(worker_dl, desc="Calculating PVE ...", total=len(worker_dl)):
                x = batch["data"]
                if x.device != device:
                    x = x.to(device, non_blocking=True)

                _, rx, _ = model(x)  # shape: (B, L, C)

                bsz = x.shape[0]  # actual batch size (last batch may be smaller)
                original_x[idx:idx + bsz].copy_(x)
                reconstructed_x[idx:idx + bsz].copy_(rx)

                idx += bsz

        # Move all tensors to CPU at once
        original_x = original_x.cpu().numpy()
        reconstructed_x = reconstructed_x.cpu().numpy()

        # Split by subject ranges
        all_original_x = []
        all_reconstructed_x = []
        for _, start, end in ranges:
            all_original_x.append(original_x[start:end].reshape(-1, n_channels))
            all_reconstructed_x.append(reconstructed_x[start:end].reshape(-1, n_channels))
            # all_*_x.shape: (N, T, C)

        pve = []
        for x, rx in tqdm(
            zip(all_original_x, all_reconstructed_x),
            desc="Calculating Percentage of Variance Explained ...",
            total=len(all_original_x),
        ):
            pve.append(
                100 * (1 - np.sum((x - rx) ** 2) / np.sum(x ** 2))
            )

        if len(pve) == 1:
            return pve[0]
        return np.array(pve)

    def get_token_kernel_response(
        self,
        dataloader: DataLoader,
        input: Optional[Union[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns stimulus response of tokens to passed input.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the input data.
        input : Union[str, np.ndarray], optional
            Input stimulus for the token kernel response.
            Can be "impulse", "tophat", or a 1D numpy array.

        Returns
        -------
        token_response : np.ndarray
            The token responses to the input stimulus.
        input : np.ndarray
            The input stimulus used to generate the token response.
        """
        _logger.info("Getting token kernel response ...")

        # Get hyperparameters
        token_dim = self.config.token_dim
        n_tokens = self.config.n_tokens

        # Refactor vocabularies
        if not self.vocab:
            _logger.info("Vocabulary is empty. Calling refactor_vocab() first.")
            self.refactor_vocab(dataloader)

        # Make a stimulus
        if input in [None, "impulse"]:
            input = np.zeros(token_dim * 2)
            input[token_dim] = 1.0
        elif input == "tophat":
            input = np.zeros(token_dim * 6)
            input[token_dim : token_dim * 5] = 1.0
        elif isinstance(input, np.ndarray) and input.ndim != 1:
            raise ValueError("Input array must be 1D.")
        else:
            raise ValueError("Unsupported input type.")
        
        n_samples = input.shape[0]

        # Get token kernel layer
        decoder_layer = self.model.decoder_layer
        token_basis_layer = decoder_layer.token_basis_layer
        device = next(token_basis_layer.parameters()).device

        # Create a single input tensor
        input_tensor = torch.tensor(
            input, dtype=torch.float32, device=device
        )  # shape: (T,)

        # Compute stimulus response for each token
        with torch.inference_mode():
            # Construct batched inputs to token kernel layer
            token_weights = torch.zeros(
                (n_tokens, n_tokens, n_samples),
                dtype=torch.float32,
                device=device,
            )
            indices = torch.arange(n_tokens, device=device)
            broadcast_tensor = input_tensor.unsqueeze(0).expand(n_tokens, -1)
            token_weights[indices, indices, :] = broadcast_tensor

            # Get kernel responses
            token_weights = decoder_layer._pad(token_weights)
            responses = token_basis_layer(token_weights).sum(dim=1)
            kernel_response = responses.cpu().numpy()  # shape: (N_t, T)

        # Remap to refactored tokens
        token_response = np.array(
            [kernel_response[order] for order in self.vocab["token_order"]]
        )  # shape: (n_refactored_tokens, T)

        return token_response, input

    # ----------------
    # Saving & Loading
    # ----------------

    def save(self, dirname: str) -> None:
        """
        Saves the model state and token vocabulary to the specified directory.

        Parameters
        ----------
        dirname : str
            Directory to save the model files.
        """
        # Save model state
        os.makedirs(dirname, exist_ok=True)
        model_path = os.path.join(dirname, "model_state.pt")
        torch.save(self.model.state_dict(), model_path)
        _logger.info(f"Saved model state to {model_path}.")

        # Save token vocabulary
        with open(os.path.join(dirname, "vocab.pkl"), "wb") as f:
            pickle.dump(self.vocab, f)
        _logger.info(f"Saved token vocabulary in {dirname}.")

    @classmethod
    def load_model(
        cls,
        dirname: str,
        config: Optional[Config] = None,
        checkpoint: Optional[str] = None,
        map_location: Optional[str] = "cpu",
        strict: Optional[bool] = True,
    ):
        """
        Loads the model from the specified directory.

        Parameters
        ----------
        dirname : str
            Directory to load the model files from.
        config : Config, optional
            Configuration object. If None, a config will be loaded from
            the specified directory.
        checkpoint : str, optional
            Checkpoint file path, file name, or "latest" to load the
            latest checkpoint. If None, the model will be loaded using a
            `model_state.pt` file.
        map_location : str, optional
            Map location for loading the model. Defaults to "cpu".
        strict : bool, optional
            Whether to enforce strict loading of model weights. Defaults to True.
        """
        # Load configuration if not provided
        if config is None:
            config = get_config(f"{dirname}/config.yml")

        # Instantiate module
        model_module = cls(config)

        # Helper function to find the latest checkpoint
        def _find_latest_ckpt(checkpoint_dir: str):
            files = sorted(
                glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime
            )
            return files[-1] if files else None

        if checkpoint:
            if checkpoint == "latest":
                ckpt_dir = os.path.join(dirname, "checkpoints")
                ckpt_path = _find_latest_ckpt(ckpt_dir)
                if ckpt_path is None:
                    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")
            elif os.path.isabs(checkpoint) or os.path.exists(checkpoint):
                ckpt_path = checkpoint
            else:
                ckpt_candidate = os.path.join(dirname, checkpoint)
                if os.path.exists(ckpt_candidate):
                    ckpt_path = ckpt_candidate
                else:
                    raise FileNotFoundError(
                        f"Checkpoint {checkpoint} not found (tried as absolute path and under {dirname})."
                    )
            _logger.info(f"Loading model from checkpoint: {ckpt_path}")

            # Load Lightning checkpoint (safe on CPU)
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            # NOTE: Includes model weights, optimizer / scheduler / AMP states, and metadata.

            # Load model weights
            state_dict = ckpt["state_dict"]
            model_module.load_state_dict(state_dict, strict=strict)

        else:
            # Weights-only path (inference-friendly)
            state_path = os.path.join(dirname, "model_state.pt")
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"Model state file not found at {state_path}.")
            _logger.info(f"Loading model from file: {state_path}")

            model_state = torch.load(state_path, map_location=map_location, weights_only=True)
            model_module.model.load_state_dict(model_state, strict=strict)

        # Load vocab if present
        vocab_path = os.path.join(dirname, "vocab.pkl")
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                model_module.vocab = pickle.load(f)

        return model_module
