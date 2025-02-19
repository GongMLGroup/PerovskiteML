import numpy as np
import pandas as pd

from .expansion import expand_dataset
from .reduction import reduce_data, partition_by_pattern
from ..utils import data_logger, setup_logger


def _to_numeric(col):
    """Internal function for converting column data to numeric values."""
    col_types = col.apply(type)
    if not np.any((col_types == bool)):
        try:
            return pd.to_numeric(col)
        except ValueError:
            return col
    return col


def preprocess_data(data, ref, threshold, depth, sections=[], exclude_cols=[], nan_equivalents={}, verbosity: int = 0):
    """Runs the entire preprocessing pipeline for the given data.

    Args:
        data (dataframe): The perovskite data.
        ref (dataframe): The reference data for features.
        threshold (float): Threshold (%) for the feature density.
            Used to remove sparce data.
        depth (float): Threshold (%) for the feature layer density.
            Determines how many feature layers are extracted.
        sections (list of str, optional): List of sections to be included.
            Defaults to [].
        exclude_cols (list of str, optional): List of columns to be excluded.
            Defaults to [].
        nan_equivalents (dict, optional): Equivalent values for NaN in the dataset.
            Defaults to {}.
        verbosity (int, optional): Verbosity level.
            Defaults to 0.

    Returns:
        dataframe: The preprocessed data.

    """
    setup_logger(verbosity)
    data_logger.info("Preproccessing Data.")
    data_logger.debug(f"Threshold: {threshold}, Depth: {depth}")
    
    patterned, nonpatterned = partition_by_pattern(ref, sections)

    patterned_data = reduce_data(data[patterned], percent=threshold)
    patterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

    nonpatterned_data = reduce_data(data[nonpatterned], percent=threshold)
    nonpatterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

    expanded_data = expand_dataset(
        patterned_data, percent=depth, verbosity=verbosity)
    
    data = pd.concat([expanded_data, nonpatterned_data], axis=1)
    data.replace(nan_equivalents, inplace=True)
    data = data.apply(_to_numeric)
    
    data_logger.info("Preprocessing Complete.")
    data_logger.info(f"Data Shape: {data.shape}")

    return data
