import os
import dotenv
import numpy as np

from .utils import PROJECT_DIR
from .data import ExpandedDataset, PerovskiteDatabase
from . import preprocessing

try:
    dotenv.load_dotenv(os.path.join(PROJECT_DIR, ".env"))
    NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
    NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
except KeyError:
    NEPTUNE_PROJECT = None
    NEPTUNE_API_TOKEN = None


__all__ = [
    "ExpandedDataset",
    "PerovskiteDatabase",
    "preprocessing",
    "NEPTUNE_PROJECT",
    "NEPTUNE_API_TOKEN"
]