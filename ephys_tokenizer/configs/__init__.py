"""Wrapper class for tokenizer configurations."""

# Import packages
from dataclasses import dataclass
from omegaconf import DictConfig
from ephys_tokenizer.configs.config import (
    EphysTokenizerConfig,
    MuTransformTokenizerConfig,
    StandardQuantileTokenizerConfig,
)
from typing import Union


CONFIGS = {
    "ephys_tokenizer": EphysTokenizerConfig,
    "mu_transform_tokenizer": MuTransformTokenizerConfig,
    "standard_quantile_tokenizer": StandardQuantileTokenizerConfig,
}


@dataclass
class Config:
    """
    Base configuration class for building and training tokenizers.
    """
    config_class: Union[
        EphysTokenizerConfig,
        MuTransformTokenizerConfig,
        StandardQuantileTokenizerConfig,
    ] = None

    def set_config(self, config: DictConfig) -> None:
        self.config_class.set_config(config)

    def validate(self) -> None:
        self.config_class.validate()


def get_config(config: DictConfig) -> Config:
    """
    Returns a Config object based on the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary containing the config.

    Returns
    -------
    cfg: Config
        Config object containing the tokenizer configuration.
    """
    # Initialize config class
    config_class = CONFIGS[config.get("name", "ephys_tokenizer")]
    cfg = Config(config_class())

    # Set model and training configurations
    cfg.set_config(config)

    # Validate configuration
    cfg.validate()

    return cfg
