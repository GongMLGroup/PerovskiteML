import tomllib
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    return config


def config_to_neptune_format(data: dict | list | Any) -> dict | list | Any:
    """Recursively convert lists to index-keyed dictionaries.

    Example:
    ["a", {"b": 2}] ➔ {"0": "a", "1": {"b": 2}}
    """
    if isinstance(data, dict):
        return {
            key: config_to_neptune_format(value)
            for key, value in data.items()
        }

    elif isinstance(data, list):
        return {
            str(index): config_to_neptune_format(item)
            for index, item in enumerate(data)
        }

    return data
