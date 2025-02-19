import os
import dotenv
import numpy as np

from .utils import PROJECT_DIR
from .data import DATABASE, DataSet

try:
    dotenv.load_dotenv(os.path.join(PROJECT_DIR, ".env"))
    NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
    NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
except KeyError:
    NEPTUNE_PROJECT = None
    NEPTUNE_API_TOKEN = None


__all__ = [
    "DATABASE",
    "DataSet",
    "NEPTUNE_PROJECT",
    "NEPTUNE_API_TOKEN"
]