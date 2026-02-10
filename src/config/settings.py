"""
Configuration management for eclipsing binary temperature analysis.

Provides centralized access to configuration parameters and paths,
eliminating hardcoded paths throughout the codebase.

Usage:
    from src.config import get_config

    config = get_config()
    data_dir = config.get_path('processed')
    missing_val = config.get('processing', 'missing_value')
"""

from pathlib import Path
from typing import Any, Optional, Dict
import yaml
import os


class Config:
    """
    Configuration manager with automatic project root detection.

    Attributes:
        project_root: Absolute path to project root directory
    """

    def __init__(self, config_path: Optional[Path] = None, project_root: Optional[Path] = None):
        """
        Initialize configuration.

        Parameters
        ----------
        config_path : Path, optional
            Path to config.yaml file. If None, searches for it automatically.
        project_root : Path, optional
            Override project root detection. If None, auto-detects.
        """
        # Find project root
        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        else:
            self.project_root = self._find_project_root()

        # Load config file
        if config_path is None:
            config_path = self.project_root / 'config' / 'config.yaml'

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Expected location: {self.project_root / 'config' / 'config.yaml'}"
            )

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def _find_project_root(self) -> Path:
        """
        Find project root by searching for .git directory or specific markers.

        Returns
        -------
        Path
            Absolute path to project root
        """
        # Start from this file's location
        current = Path(__file__).resolve().parent

        # Search up the directory tree
        while current != current.parent:
            # Look for project markers
            if (current / '.git').exists():
                return current
            if (current / 'config' / 'config.yaml').exists():
                return current
            if (current / 'setup.py').exists() or (current / 'pyproject.toml').exists():
                return current
            current = current.parent

        # If no markers found, use current working directory
        # or fall back to three levels up from this file (src/config/settings.py)
        fallback = Path(__file__).resolve().parent.parent.parent
        return fallback

    def get_path(self, key: str, ensure_exists: bool = False) -> Path:
        """
        Get absolute path from configuration.

        Parameters
        ----------
        key : str
            Key in the 'paths' section of config
        ensure_exists : bool, default False
            If True, create directory if it doesn't exist

        Returns
        -------
        Path
            Absolute path

        Examples
        --------
        >>> config = get_config()
        >>> data_dir = config.get_path('processed')
        >>> models_dir = config.get_path('models', ensure_exists=True)
        """
        if key not in self._config['paths']:
            raise KeyError(f"Path key '{key}' not found in configuration")

        rel_path = self._config['paths'][key]
        abs_path = self.project_root / rel_path

        if ensure_exists:
            abs_path.mkdir(parents=True, exist_ok=True)

        return abs_path

    def get_dataset_path(self, dataset_key: str, location: str = 'processed') -> Path:
        """
        Get path to a specific dataset file.

        Parameters
        ----------
        dataset_key : str
            Key in the 'datasets' section of config
        location : str, default 'processed'
            Which data directory ('raw', 'processed', 'external', 'interim')

        Returns
        -------
        Path
            Absolute path to dataset file

        Examples
        --------
        >>> config = get_config()
        >>> catalog_path = config.get_dataset_path('eb_catalog', 'raw')
        >>> ml_data_path = config.get_dataset_path('ml_training_clean')
        """
        if dataset_key not in self._config['datasets']:
            raise KeyError(f"Dataset key '{dataset_key}' not found in configuration")

        filename = self._config['datasets'][dataset_key]
        location_path = self.get_path(location)

        return location_path / filename

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value.

        Parameters
        ----------
        *keys : str
            Sequence of keys to navigate nested config
        default : Any, optional
            Default value if key path not found

        Returns
        -------
        Any
            Configuration value

        Examples
        --------
        >>> config = get_config()
        >>> missing_val = config.get('processing', 'missing_value')
        >>> test_size = config.get('ml', 'test_size')
        """
        try:
            value = self._config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key path {' -> '.join(keys)} not found")

    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.

        Returns
        -------
        dict
            Complete configuration
        """
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config(project_root={self.project_root})"


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(reload: bool = False, config_path: Optional[Path] = None) -> Config:
    """
    Get singleton configuration instance.

    Parameters
    ----------
    reload : bool, default False
        Force reload configuration from file
    config_path : Path, optional
        Custom config file path (only used on first load or reload)

    Returns
    -------
    Config
        Configuration instance

    Examples
    --------
    >>> from src.config import get_config
    >>> config = get_config()
    >>> data_dir = config.get_path('processed')
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config(config_path=config_path)

    return _config_instance


def reset_config():
    """Reset configuration singleton (useful for testing)."""
    global _config_instance
    _config_instance = None
