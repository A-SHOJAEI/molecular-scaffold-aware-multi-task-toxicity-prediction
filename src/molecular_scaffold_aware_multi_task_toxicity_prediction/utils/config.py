"""Configuration management and utility functions."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration management class with hierarchical parameter loading.

    Supports loading from YAML files with environment variable overrides
    and provides type-safe parameter access.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.

        Args:
            config_dict: Dictionary with configuration parameters
        """
        self._config = config_dict or {}

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {config_path}")
            return cls(config_dict)

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            logger.info(f"Loaded JSON configuration from {config_path}")
            return cls(config_dict)

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing JSON config: {e}", e.doc, e.pos) from e

    def get(self,
            key: str,
            default: Any = None,
            required: bool = False) -> Any:
        """Get configuration parameter with support for nested keys.

        Args:
            key: Parameter key (supports dot notation, e.g., 'model.hidden_dim')
            default: Default value if key not found
            required: Whether parameter is required

        Returns:
            Parameter value

        Raises:
            KeyError: If required parameter is missing
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value

        except (KeyError, TypeError):
            if required:
                raise KeyError(f"Required configuration parameter missing: {key}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration parameter.

        Args:
            key: Parameter key (supports dot notation)
            value: Parameter value
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, other: Union['Config', Dict[str, Any]]) -> None:
        """Update configuration with parameters from another config.

        Args:
            other: Another Config instance or dictionary
        """
        if isinstance(other, Config):
            other_dict = other._config
        else:
            other_dict = other

        self._deep_update(self._config, other_dict)

    def _deep_update(self, base: Dict, update: Dict) -> None:
        """Recursively update nested dictionary.

        Args:
            base: Base dictionary to update
            update: Updates to apply
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {path}")

    def __getitem__(self, key: str) -> Any:
        """Get configuration parameter using bracket notation.

        Args:
            key: Parameter key

        Returns:
            Parameter value
        """
        return self.get(key, required=True)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration parameter using bracket notation.

        Args:
            key: Parameter key
            value: Parameter value
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key.

        Args:
            key: Parameter key

        Returns:
            True if key exists
        """
        try:
            self.get(key, required=True)
            return True
        except KeyError:
            return False

    def __repr__(self) -> str:
        """String representation of configuration.

        Returns:
            String representation
        """
        return f"Config({self._config})"


def load_config(config_path: Union[str, Path],
                override_dict: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration with optional overrides from YAML or JSON files.

    Args:
        config_path: Path to YAML or JSON configuration file
        override_dict: Optional dictionary to override parameters

    Returns:
        Loaded configuration

    Raises:
        ValueError: If file extension is not supported
    """
    config_path = Path(config_path)

    # Determine file type by extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        config = Config.from_yaml(config_path)
    elif config_path.suffix.lower() == '.json':
        config = Config.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. "
                        f"Supported formats: .yaml, .yml, .json")

    # Apply environment variable overrides
    env_overrides = {}
    for key, value in os.environ.items():
        if key.startswith('TOXICITY_'):
            config_key = key[9:].lower().replace('_', '.')  # Remove TOXICITY_ prefix

            # Try to parse as number or boolean
            try:
                if '.' in value:
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    parsed_value = value.lower() == 'true'
                else:
                    parsed_value = value

            env_overrides[config_key] = parsed_value

    if env_overrides:
        config.update(env_overrides)
        logger.info(f"Applied environment overrides: {list(env_overrides.keys())}")

    # Apply explicit overrides
    if override_dict:
        config.update(override_dict)
        logger.info(f"Applied explicit overrides: {list(override_dict.keys())}")

    return config


def get_device(config: Optional[Config] = None) -> torch.device:
    """Get the appropriate device for training.

    Args:
        config: Optional configuration object

    Returns:
        PyTorch device
    """
    if config and 'device' in config:
        device_str = config['device']
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device.index)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU")

    return device


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducible results.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on GPU (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seeds to {seed}")


def setup_logging(config: Optional[Config] = None) -> None:
    """Setup logging configuration.

    Args:
        config: Optional configuration object
    """
    log_level = 'INFO'
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if config:
        log_level = config.get('logging.level', 'INFO').upper()
        log_format = config.get('logging.format', log_format)

    # Setup root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )

    # Setup file logging if specified
    if config and 'logging.file' in config:
        log_file = Path(config['logging.file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter(log_format))

        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # Suppress verbose third-party loggers
    verbose_loggers = [
        'urllib3.connectionpool',
        'matplotlib.font_manager',
        'PIL.PngImagePlugin',
    ]

    for logger_name in verbose_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_model_config(config: Config, model_name: str) -> Dict[str, Any]:
    """Extract model-specific configuration.

    Args:
        config: Main configuration object
        model_name: Name of model ('gcn', 'gat', 'sage')

    Returns:
        Model configuration dictionary

    Raises:
        KeyError: If model configuration is missing
    """
    # Add common model parameters
    common_params = {
        'node_dim': config.get('data.node_dim', required=True),
        'hidden_dim': config.get('model.hidden_dim', 128),
        'num_layers': config.get('model.num_layers', 3),
        'dropout': config.get('model.dropout', 0.2),
        'scaffold_dim': config.get('model.scaffold_dim', 64),
    }

    # Add model-specific parameters that the model constructor accepts
    if model_name == 'gat':
        common_params['num_heads'] = config.get('model.gat.num_heads', 4)
    elif model_name == 'sage':
        common_params['aggr'] = config.get('model.sage.aggregation', 'mean')

    return common_params


def create_experiment_name(config: Config) -> str:
    """Create descriptive experiment name from configuration.

    Args:
        config: Configuration object

    Returns:
        Experiment name string
    """
    model_name = config.get('model.name', 'unknown')
    dataset = config.get('data.dataset', 'unknown')
    timestamp = torch.cuda.current_device() if torch.cuda.is_available() else 0

    experiment_name = f"toxicity_{model_name}_{dataset}"

    # Add key hyperparameters
    if 'model.hidden_dim' in config:
        experiment_name += f"_h{config['model.hidden_dim']}"

    if 'training.learning_rate' in config:
        lr = config['training.learning_rate']
        experiment_name += f"_lr{lr:.0e}"

    return experiment_name


def validate_config(config: Config) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    logger.debug("Validating configuration parameters")

    required_params = [
        'model.name',
        'data.dataset',
        'training.num_epochs',
        'training.batch_size',
        'training.learning_rate',
    ]

    missing_params = []
    for param in required_params:
        if param not in config:
            missing_params.append(param)

    if missing_params:
        logger.error(f"Configuration validation failed: missing required parameters: {missing_params}")
        raise ValueError(f"Required parameters missing: {', '.join(missing_params)}")

    logger.debug(f"All {len(required_params)} required parameters present")

    # Validate model name
    valid_models = ['gcn', 'gat', 'sage']
    model_name = config.get('model.name')
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {valid_models}")

    # Validate dataset
    valid_datasets = ['tox21', 'toxcast', 'clintox']
    dataset = config.get('data.dataset')
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Must be one of {valid_datasets}")

    # Validate numeric parameters
    numeric_params = {
        'training.num_epochs': (1, 1000),
        'training.batch_size': (1, 1024),
        'training.learning_rate': (1e-6, 1.0),
        'model.hidden_dim': (8, 2048),
        'model.num_layers': (1, 10),
        'model.dropout': (0.0, 1.0),
    }

    for param, (min_val, max_val) in numeric_params.items():
        if param in config:
            value = config[param]
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter {param}={value} outside valid range [{min_val}, {max_val}]")

    logger.info("Configuration validation passed")


class ConfigDict(dict):
    """Dictionary subclass with dot notation access.

    Allows accessing nested dictionary values using dot notation
    for convenience in configuration handling.
    """

    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation.

        Args:
            name: Attribute name

        Returns:
            Attribute value

        Raises:
            AttributeError: If attribute not found
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation.

        Args:
            name: Attribute name
            value: Attribute value
        """
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete attribute using dot notation.

        Args:
            name: Attribute name
        """
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")