import numpy as np
import pandas as pd

from .expansion import expand_dataset
from .reduction import reduce_data, partition_by_pattern


def _to_numeric(col):
    """Internal function for converting column data to numeric values."""
    col_types = col.apply(type)
    if not np.any((col_types == bool)):
        return pd.to_numeric(col, errors='ignore')
    return col


def preprocess_data(data, ref, threshold, depth, sections=[], exclude_cols=[], nan_equivalents={}, verbose: bool = True):
    """Runs the entire preprocessing pipeline for the dataset.

    Args:
        target (str): Name of the target feature
        threshold (float): Threshold (%) for the feature density.
            Used to remove sparce data.
        depth (float): Threshold (%) for the feature layer density.
            Determines how many feature layers are extracted.
        exclude_sections (list of str, optional): List of sections to be excluded.
            Defaults to [].
        exclude_cols (list of str, optional): List of columns to be excluded.
            Defaults to [].

    Returns:
        dataframe: The preprocessed data.
        series: The target data.

    """
    patterned, nonpatterned = partition_by_pattern(ref, sections)

    patterned_data = reduce_data(data[patterned], percent=threshold)
    patterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

    nonpatterned_data = reduce_data(data[nonpatterned], percent=threshold)
    nonpatterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

    expanded_data = expand_dataset(
        patterned_data, percent=depth, verbose=verbose)
    print("Data Preprocessed.")
    data = pd.concat([expanded_data, nonpatterned_data], axis=1)
    data.replace(nan_equivalents, inplace=True)
    data = data.apply(_to_numeric)

    return data
