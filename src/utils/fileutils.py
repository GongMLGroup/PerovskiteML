import os
import hashlib

def _updir(path, depth=1):
    """Finds the directory `depth` number of steps above the given path."""
    for _ in range(depth):
        path = os.path.dirname(path)
    return path

def hash_params(params: dict):
    """Encodes preprocessing parameters into a hash. Is used to create unique file names for preprocessed data.

    Args:
        params (dict): The preprocessing parameters.

    Returns:
        str: The hash of the parameters.

    """
    param_str = '_'.join(f'{key}={value}' for key,
                         value in sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

PROJECT_DIR =  _updir(__file__, 3)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
EXPAND_DIR = os.path.join(DATA_DIR, 'expanded')
