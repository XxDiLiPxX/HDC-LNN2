import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """Configuration class that loads settings from YAML files."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            val = self._data[item]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"No such configuration key: {item}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

def load_config(base_path: Path, override_path: Optional[Path] = None) -> Config:
    """Loads a base configuration YAML and merges it with an optional override configuration."""
    with open(base_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f) or {}

    if override_path and override_path.exists():
        with open(override_path, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f) or {}
        _deep_update(config_data, overrides)

    return Config(config_data)

def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively updates base dictionary with overrides in-place."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
