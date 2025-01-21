import os
import dotenv
import json
import numpy as np

from .fileutils import PROJECT_DIR
from .database import PerovskiteData

dotenv.load_dotenv(os.path.join(PROJECT_DIR, ".env"))
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

with open(os.path.join(PROJECT_DIR,"data/section_keys.json"), "r") as file:
    SECTION_KEYS = json.load(file)
    """dict: Dictionary of section names and their corresponding shorthand.

    The shorhand is used as a prefix for feature names.
    """
with open(os.path.join(PROJECT_DIR,"data/nan_equivalents.json"), "r") as file:
    NAN_EQUIVALENTS = json.load(file)
    """dict: Keys are equivalent to `missing` or `nan` values in the dataset."""

DATASET = PerovskiteData(
    ref_file="pdp_units_data.xlsx",
    database_file="Perovskite_database.csv",
    nan_equivalents=NAN_EQUIVALENTS,
    section_keys=SECTION_KEYS
)
"""An instance of the Perovskite Dataset."""


def display_scores(scores):
    """Print formatted Mean and Std"""
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(
        scores, np.mean(scores), np.std(scores)))
