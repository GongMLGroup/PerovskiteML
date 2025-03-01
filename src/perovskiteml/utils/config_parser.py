import tomllib
from pathlib import Path
from typing import Union


def load_config(config_path: Union[str, Path]):
    config_path = Path(config_path)
    
    with config_path.open("rb") as file:
        config = tomllib.load(file)
        
    return config