"""Configuration classes for tokenizers."""

# Import packages
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "name": "adam",
        "learning_rate": 1e-3,
        "eps": 1e-7,
    })
    batch_size: int = 32
    n_epochs: int = 10
    multi_gpu: bool = False
    callbacks: Optional[List[Any]] = field(default_factory=list)


@dataclass
class CallbackConfig:
    temperature_annealing: Optional[Dict[str, Any]] = None


@dataclass
class EphysTokenizerConfig:
    """
    Data-driven tokenizer based on the standard autoencoder architecture.
    """
    name: str = "ephys_tokenizer"

    # Base defaults
    sequence_length: Optional[int] = None
    n_channels: Optional[int] = None

    # Tokenizer defaults
    n_tokens: int = 256
    token_dim: int = 32
    token_kernel_padding: str = "same"
    token_kernel_bias: bool = True
    token_groups: int = 1

    # RNN defaults
    rnn_n_units: int = 64
    rnn_type: str = "gru"
    rnn_n_layers: int = 1

    # Training defaults
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Callback defaults
    callback: CallbackConfig = field(default_factory=CallbackConfig)

    def validate(self) -> None:
        self.validate_model_config()
        self.validate_training_config()

    def validate_model_config(self) -> None:
        assert self.sequence_length is not None, "sequence_length must be set"
        assert self.n_channels is not None, "n_channels must be set"
        assert self.n_tokens > 0, "n_tokens must be greater than 0"
        assert self.token_dim > 0, "token_dim must be greater than 0"
        assert self.token_kernel_padding in ["same", "causal"], \
            "token_kernel_padding must be 'same' or 'causal'"
        assert self.token_kernel_bias is not None, "token_kernel_bias must be set"
        assert self.token_groups > 0, "token_groups must be greater than 0"
        assert self.rnn_n_units > 0, "rnn_n_units must be greater than 0"
        assert self.rnn_type in ["gru", "lstm"], "rnn_type must be 'gru' or 'lstm'"
        assert self.rnn_n_layers > 0, "rnn_n_layers must be greater than 0"

    def validate_training_config(self) -> None:
        assert self.training.optimizer is not None, "optimizer must be set"
        assert self.training.batch_size > 0, "batch_size must be greater than 0"
        assert self.training.n_epochs > 0, "n_epochs must be greater than 0"

    def set_config(self, config: DictConfig) -> None:
        self.sequence_length = config.get("sequence_length", self.sequence_length)
        self.n_channels = config.get("n_channels", self.n_channels)
        self.n_tokens = config.get("n_tokens", self.n_tokens)
        self.token_dim = config.get("token_dim", self.token_dim)
        self.token_kernel_padding = config.get("token_kernel_padding", self.token_kernel_padding)
        self.rnn_n_units = config.get("rnn_n_units", self.rnn_n_units)
        self.rnn_type = config.get("rnn_type", self.rnn_type)
        self.rnn_n_layers = config.get("rnn_n_layers", self.rnn_n_layers)

        self._set_training_config(config.get("training", self.training))
        self._set_callback_config(config.get("callback", self.callback))

    def _set_training_config(self, config: DictConfig) -> None:
        if config is None:
            return

        self.training = OmegaConf.merge(
            OmegaConf.structured(self.training), config
        )

    def _set_callback_config(self, config: DictConfig) -> None:
        if config is None:
            return

        if "temperature_annealing" in config:
            default_ta_dict = {
                "n_stages": 10,
                "n_epochs": 10,
                "start_temperature": 1.0,
                "end_temperature": 1e-3,
                "n_annealing_epochs": 10,
            }
            config.temperature_annealing = config.get("temperature_annealing", default_ta_dict)

        self.callback = OmegaConf.merge(
            OmegaConf.structured(self.callback), config
        )


@dataclass
class MuTransformTokenizerConfig:
    """
    Non-learnable tokenizer with mu-transformation and quantized binning.

    Reference:
        - WaveNet: A generative model for raw audio (van den Oord et al., 2016)
    """
    name: str = "mu_transform_tokenizer"
    n_tokens: Optional[int] = None
    mu: Optional[int] = None
    normalization: str = "max_abs"

    def validate(self) -> None:
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.n_tokens > 0, "n_tokens must be greater than 0"
        assert self.mu is not None, "mu must be set"
        assert self.mu > 0, "mu must be greater than 0"
        assert self.normalization in ["max_abs", "min_max"], \
        "normalization must be 'max_abs' or 'min_max'"

    def set_config(self, config: DictConfig) -> None:
        self.n_tokens = config.get("n_tokens", 256)
        self.mu = config.get("mu", 255)
        if "normalization" in config:  # allow override
            self.normalization = str(config["normalization"])


@dataclass
class StandardQuantileTokenizerConfig:
    """
    Non-learnable tokenizer with z-normalization scaling and 
    quantile-based binning.

    Reference:
        - Chronos: Learning the Language of Time Series (Ansari et al., 2024)
    """
    name: str = "standard_quantile_tokenizer"
    n_tokens: Optional[int] = None
    standardize: bool = True

    def validate(self) -> None:
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.n_tokens > 0, "n_tokens must be greater than 0"

    def set_config(self, config: DictConfig) -> None:
        self.n_tokens = config.get("n_tokens", 256)
        if "standardize" in config:  # allow override
            self.standardize = config["standardize"]
