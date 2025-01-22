import os
import dotenv
import numpy as np

from .fileutils import PROJECT_DIR
from .database import DATABASE
from .dataset import DataSet

dotenv.load_dotenv(os.path.join(PROJECT_DIR, ".env"))
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")


def display_scores(scores):
    """Print formatted Mean and Std"""
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(
        scores, np.mean(scores), np.std(scores)))
