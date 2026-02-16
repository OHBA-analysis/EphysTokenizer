"""
Layers for the tokenizer models.

Mathematical Notation:
  - B: batch size
  - L: sequence length
  - C: number of channels
  - N_t: number of tokens
"""

# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple


def rnn_layer(
    rnn_type: str, input_size: int, hidden_size: int
) -> nn.Module:
    """
    Creates a single-layer GRU/LSTM.

    Parameters
    ----------
    rnn_type : str
        Type of an RNN layer. Options include 'gru' and 'lstm'.
    input_size : int
        The number of expected features in the input.
    hidden_size : int
        The number of features in the hidden state.

    Returns
    -------
    rnn_module : nn.Module
        The RNN layer.
    """
    if rnn_type == "gru":
        rnn_module = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
    elif rnn_type == "lstm":
        rnn_module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")
    return rnn_module


class TokenWeightsLayer(nn.Module):
    """
    A layer that projects encoder outputs to per-token weights
    with annealed sampling strategy.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    output_dim : int
        The number of output features.
    activation : str
        The activation function to use. Defaults to "linear".
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "linear",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.dense_layer = nn.Linear(input_dim, output_dim)
        self.activation_fn = self._get_activation_fn(activation)
        self.norm_layer = nn.LayerNorm(output_dim)
        
        # Register temperature buffer (float scalar)
        self.register_buffer("_temperature", torch.tensor(0.0))

    def _get_activation_fn(self, activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "linear":
            return nn.Identity()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    @property
    def temperature(self) -> float:
        return float(self._temperature.item())

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature.fill_(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Project to token logits
        ell = self.activation_fn(self.dense_layer(inputs))
        ell = self.norm_layer(ell) / 0.1  # normalize and scale
        # shape: (B * C, L, N_t)

        if self.training:
            # Hard one-hot samples using argmax
            theta_sample_idx = torch.argmax(ell, dim=2)
            theta_sample = F.one_hot(
                theta_sample_idx, num_classes=self.output_dim
            ).to(ell.dtype)

            # Soft weights via softmax
            theta_weight = F.softmax(ell, dim=2)

            # Perform annealing between soft and hard samples
            token_weight = (
                self.temperature * theta_weight + (1.0 - self.temperature) * theta_sample
            )
        else:
            # If not training, use hard argmax one-hot
            token_weight = F.one_hot(
                torch.argmax(ell, dim=2), num_classes=self.output_dim
            ).to(ell.dtype)  # shape: (B * C, L, N_t)

        return token_weight


class MSELossLayer(nn.Module):
    """
    A loss layer for the mean squared error objective.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean squared error loss.

        Parameters
        ----------
        y_true : torch.Tensor
            The ground truth tensor.
            Shape is (batch_size, sequence_length, n_channels).
        y_pred : torch.Tensor
            The predicted tensor.
            Shape is (batch_size, sequence_length, n_channels).

        Returns
        -------
        loss : torch.Tensor
            The computed loss. Shape is (1,).
        """
        return torch.mean((y_true - y_pred) ** 2)


class EncoderLayer(nn.Module):
    """
    Encodes input multi-variate time-series with RNN(s).

    Parameters
    ----------
    rnn_type : str
        Type of the RNN layer. Either 'gru' or 'lstm'.
    rnn_n_layers : int
        Number of layers in the RNN.
    rnn_n_units : int
        Number of units in the RNN.
    """
    def __init__(self, rnn_type: str, rnn_n_layers: int, rnn_n_units: int):
        super().__init__()
        self.rnn_type = rnn_type
        self.rnn_n_layers = rnn_n_layers
        self.rnn_n_units = rnn_n_units

        # Build a ModuleList of RNN layers
        self.rnn_layers = nn.ModuleList(
            [
                rnn_layer(
                    rnn_type=rnn_type,
                    input_size=rnn_n_units if i > 0 else 1,
                    hidden_size=rnn_n_units,
                )
                for i in range(rnn_n_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Prepare inputs
        x = inputs  # shape: (B, L, C)

        # Transpose dimensions
        x = x.permute(0, 2, 1)  # shape: (B, C, L)

        # Reshape for RNN
        x = x.reshape(-1, x.shape[-1], 1)  # shape: (B * C, L, 1)

        # Pass through the encoder (stacked RNNs)
        for rnn in self.rnn_layers:
            x, _ = rnn(x)  # shape: (B * C, L, rnn_n_units)

        return x


class DecoderLayer(nn.Module):
    """
    Decodes token weights back into the multi-variate time-series.

    Parameters
    ----------
    n_channels : int
        Number of channels in the data.
    sequence_length : int
        Length of the sequence.
    n_tokens : int
        Number of tokens.
    token_dim : int
        Kernel window size.
    token_kernel_padding : str
        Padding strategy for the token convolutional layers.
        Should be either 'same' or 'causal'.
    """
    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_tokens: int,
        token_dim: int,
        token_kernel_padding: str,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.token_kernel_padding = token_kernel_padding

        self.token_basis_layer = nn.Conv1d(
            in_channels=self.n_tokens,
            out_channels=self.n_tokens,
            kernel_size=self.token_dim,
            padding=0,  # applied manually (matches TF "same"/"causal")
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        k = self.token_dim
        if self.token_kernel_padding == "same":
            left = (k - 1) // 2
            right = (k - 1) - left
            return F.pad(x, (left, right))
        elif self.token_kernel_padding == "causal":
            return F.pad(x, (k - 1, 0))
        else:
            raise ValueError(f"Unknown token kernel padding: {self.token_kernel_padding}")

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare inputs
        x = inputs.permute(0, 2, 1)  # shape: (B * C, N_t, L)

        # Apply 1D convolutions
        x = self._pad(x)
        x = self.token_basis_layer(x)
        x = x.sum(dim=1)  # sum across token dimension
        # shape: (B * C, L)

        # Reshape back to original dimensions
        x = x.reshape(-1, self.n_channels, self.sequence_length)
        x = x.permute(0, 2, 1)  # shape: (B, L, C)

        # Reshape inputs as token weights
        token_weights = torch.reshape(
            inputs,
            shape=(-1, self.n_channels, self.sequence_length, self.n_tokens),
        )
        token_weights = token_weights.permute(0, 2, 1, 3)  # shape: (B, L, C, N_t)

        return x, token_weights
