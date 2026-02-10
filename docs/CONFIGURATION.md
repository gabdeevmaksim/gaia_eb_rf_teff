# Configuration System

## Overview

The project uses a centralized configuration system to eliminate hardcoded paths and make the codebase portable and maintainable.

## Configuration File

All configuration is stored in `config/config.yaml`. This file contains:

- **Paths**: Relative paths to data directories
- **Datasets**: Filenames for various datasets
- **Processing parameters**: Missing values, thresholds, etc.
- **ML parameters**: Model hyperparameters
- **Temperature coefficients**: Empirical color-temperature relations

## Usage in Scripts

### Basic Usage

```python
from src.config import get_config

# Get configuration instance
config = get_config()

# Get absolute paths
data_dir = config.get_path('processed')
models_dir = config.get_path('models', ensure_exists=True)  # Creates if doesn't exist

# Get dataset paths (input in raw, output in processed)
input_photometry = config.get_dataset_path('eb_unified_photometry', 'raw')
catalog_with_teff = config.get_dataset_path('eb_catalog_teff', 'processed')

# Get configuration values
missing_val = config.get('processing', 'missing_value')
test_size = config.get('ml', 'test_size')
```

### Project Root Detection

The configuration system automatically finds the project root by searching for:
1. `.git` directory
2. `config/config.yaml` file
3. `setup.py` or `pyproject.toml`

This means scripts work regardless of where they're executed from.

### Example: Converting a Script

**Before (hardcoded paths):**
```python
from pathlib import Path

def main():
    data_dir = Path("data/external")
    input_file = data_dir / "input.csv"
    output_file = data_dir / "output.csv"
    missing_val = -999.0

    # ... processing ...
```

**After (using config):**
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

def main():
    config = get_config()

    input_file = config.get_dataset_path('eb_unified_photometry', 'raw')
    output_file = config.get_dataset_path('eb_catalog_teff', 'processed')
    missing_val = config.get('processing', 'missing_value')

    # ... processing ...
```

## Configuration API

### `get_config(reload=False, config_path=None)`

Get singleton configuration instance.

**Parameters:**
- `reload` (bool): Force reload configuration
- `config_path` (Path, optional): Custom config file path

**Returns:** `Config` instance

### `Config.get_path(key, ensure_exists=False)`

Get absolute path from configuration.

**Parameters:**
- `key` (str): Key in 'paths' section
- `ensure_exists` (bool): Create directory if it doesn't exist

**Returns:** `Path` object

**Available keys:**
- `data_root`, `raw`, `processed`, `external`, `interim`, `cache`
- `models`, `reports`, `figures`

### `Config.get_dataset_path(dataset_key, location='processed')`

Get path to a specific dataset file.

**Parameters:**
- `dataset_key` (str): Key in 'datasets' section
- `location` (str): Data directory ('raw', 'processed', 'external', 'interim')

**Returns:** `Path` object

### `Config.get(*keys, default=None)`

Get nested configuration value.

**Parameters:**
- `*keys` (str): Sequence of keys
- `default` (Any): Default value if not found

**Returns:** Configuration value

**Examples:**
```python
config.get('processing', 'missing_value')  # -999.0
config.get('ml', 'test_size')  # 0.2
config.get('temperature', 'gr_coefficients', 'A')  # 1.09
```

### `Config.project_root`

Absolute path to project root directory (auto-detected).

## Adding New Configuration

To add new datasets or parameters:

1. Edit `config/config.yaml`
2. Add entries in appropriate section
3. Use in scripts via `config.get()` or `config.get_dataset_path()`

**Example - Adding a new dataset:**

```yaml
datasets:
  # ... existing datasets ...
  my_new_data: my_new_file.parquet
```

```python
# In script
new_data_path = config.get_dataset_path('my_new_data', 'processed')
```

## Benefits

1. **Portability**: Works on any machine without path modifications
2. **Maintainability**: Single place to update paths
3. **Flexibility**: Easy to reorganize directory structure
4. **Testing**: Can use different configs for testing
5. **Documentation**: Configuration is self-documenting in YAML

## Testing

For testing with custom configuration:

```python
from src.config import get_config, reset_config

# Use custom config
config = get_config(reload=True, config_path=Path('test_config.yaml'))

# Reset to default
reset_config()
config = get_config()
```
