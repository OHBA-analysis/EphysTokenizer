"""Configuration classes for tokenizers."""

# Import packages
import yaml
from dataclasses import dataclass, field
from typing import Any, Optional, Union


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

    # RNN defaults
    rnn_n_units: int = 64
    rnn_type: str = "gru"
    rnn_n_layers: int = 1

    # Training defaults
    optimizer: Optional[Any] = None
    batch_size: int = 32
    n_epochs: int = 10
    multi_gpu: bool = False
    callbacks: Optional[list] = field(default_factory=list)

    # Callback defaults
    temperature_annealing: Optional[dict] = None

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
        assert self.rnn_n_units > 0, "rnn_n_units must be greater than 0"
        assert self.rnn_type in ["gru", "lstm"], "rnn_type must be 'gru' or 'lstm'"
        assert self.rnn_n_layers > 0, "rnn_n_layers must be greater than 0"

    def validate_training_config(self) -> None:
        assert self.optimizer is not None, "optimizer must be set"
        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.n_epochs > 0, "n_epochs must be greater than 0"

    def set_config(self, config: dict) -> None:
        self.set_model_config(config)
        self.set_training_config(config)
        self.set_callback_config(config)

    def set_model_config(self, config: dict) -> None:
        self.sequence_length = config.get("sequence_length", None)
        self.n_channels = config.get("n_channels", None)
        self.n_tokens = config.get("n_tokens", self.n_tokens)
        self.token_dim = config.get("token_dim", self.token_dim)
        self.token_kernel_padding = config.get("token_kernel_padding", self.token_kernel_padding)
        self.rnn_n_units = config.get("rnn_n_units", self.rnn_n_units)
        self.rnn_type = config.get("rnn_type", self.rnn_type)
        self.rnn_n_layers = config.get("rnn_n_layers", self.rnn_n_layers)

    def set_training_config(self, config: dict) -> None:
        default_optim_dict = {
            "name": "adam",
            "learning_rate": 1e-3,
            "eps": 1e-7,
        }
        self.optimizer = config.get("optimizer", default_optim_dict)
        self.batch_size = config.get("batch_size", self.batch_size)
        self.n_epochs = config.get("n_epochs", self.n_epochs)
        self.multi_gpu = config.get("multi_gpu", self.multi_gpu)

    def set_callback_config(self, config: dict) -> None:
        if "temperature_annealing" in config:
            default_ta_dict = {
                "n_stages": 10,
                "n_epochs": 10,
                "start_temperature": 1.0,
                "end_temperature": 1e-3,
                "n_annealing_epochs": 10,
            }
            self.temperature_annealing = config.get("temperature_annealing", default_ta_dict)


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

    def set_config(self, config: dict) -> None:
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

    def set_config(self, config: dict) -> None:
        self.n_tokens = config.get("n_tokens", 256)
        if "standardize" in config:  # allow override
            self.standardize = config["standardize"]


def load_config(config: Union[str, dict]) -> dict:
    """
    Loads configuration dictionary.

    Parameters
    ----------
    config : Union[str, dict]
        Path to a yaml file, string to convert to dictionary,
        or dictionary containing the config.

    Returns
    -------
    config : dict
        Config object for a full pipeline.
    """
    # Check the input argument type
    if type(config) not in [str, dict]:
        raise TypeError(
            f"config must be a str or dict, got {type(config)}."
        )

    # Get configuration
    if isinstance(config, str):
        try:
            # Check if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # If we have a string, load it directly
            config = yaml.load(config, Loader=yaml.FullLoader)

    if config is None:
        return {}
    return config
