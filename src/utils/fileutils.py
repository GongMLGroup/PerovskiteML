import os

def _updir(path, depth=1):
    for _ in range(depth):
        path = os.path.dirname(path)
    return path

PROJECT_DIR =  _updir(__file__, 3)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
