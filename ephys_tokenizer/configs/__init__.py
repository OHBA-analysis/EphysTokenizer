"""Wrapper class for tokenizer configurations."""

# Import packages
import yaml
from dataclasses import dataclass
from typing import Union
from ephys_tokenizer.configs.config import (
    load_config,
    EphysTokenizerConfig,
    MuTransformTokenizerConfig,
    StandardQuantileTokenizerConfig,
)


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

    def validate(self) -> None:
        self.config_class.validate()

    def set_config(self, config: dict) -> None:
        self.config_class.set_config(config)

    def get_config(self) -> dict:
        return self.config_class.to_dict()

    def save_config(self, dirname: str) -> None:
        with open(f"{dirname}/config.yml", "w") as f:
            yaml.dump(self.config_class.to_dict(), f)


def get_config(config: Union[str, dict]) -> Config:
    """
    Returns a Config object based on the provided configuration.

    Parameters
    ----------
    config : Union[str, dict]
        Path to a yaml file, string to convert to dictionary,
        or dictionary containing the config.

    Returns
    -------
    cfg: Config
        Config object containing the tokenizer configuration.
    """
    # Load configuration dictionary
    config_dict = load_config(config)

    # Initialize config class
    config_class = CONFIGS[config_dict["name"]]
    cfg = Config(config_class())

    # Set model and training configurations
    cfg.set_config(config_dict)

    # Validate configuration
    cfg.validate()

    return cfg
