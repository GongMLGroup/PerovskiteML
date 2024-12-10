import os
import copy
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .expansion import expand_dataset
from .reduction import reduce_data, partition_by_pattern
from .fileutils import DATA_DIR


def _to_numeric(col):
    """Internal function for converting column data to numeric values."""
    col_types = col.apply(type)
    if not np.any((col_types == bool)):
        return pd.to_numeric(col, errors='ignore')
    return col


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


def preprocess_data(target, threshold, depth, exclude_sections=[], exclude_cols=[], verbose: bool = True):
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
    global SECTION_KEYS
    global DATASET
    DATASET.load_data()

    params = {
        'target': target,
        'threshold': threshold,
        'depth': depth,
        'xsections': exclude_sections,
        'xcols': exclude_cols,
    }

    # Generate file name from parameters
    file_hash = hash_params(params)
    file_name = f'expanded_data_d{depth}_t{threshold}_{file_hash}.parquet'
    folder_path = os.path.join(DATA_DIR, 'preprocessed')
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif os.path.exists(file_path):
        # Check if file exists
        print(f'File already exists: {file_path}')
        print('Loading Data...')
        table = pq.read_table(file_path)
        return DATASET.get_Xy(table.to_pandas(), target)

    print(f"File does not exist, preprocessing and saving to {file_path}")
    print("Preprocessing Data...")
    print(f"Threshold: {threshold}, Depth: {depth}")
    data = DATASET.data
    ref = DATASET.ref

    # Remove excluded keys
    keys = copy.copy(SECTION_KEYS)
    for key in exclude_sections:
        del keys[key]

    patterned, nonpatterned = partition_by_pattern(ref, keys)

    patterned_data = reduce_data(data[patterned], percent=threshold)
    patterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

    nonpatterned_data = reduce_data(data[nonpatterned], percent=threshold)
    nonpatterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

    expanded_data = expand_dataset(
        patterned_data, percent=depth, verbose=verbose)
    print("Data Preprocessed.")
    data = pd.concat([expanded_data, nonpatterned_data], axis=1)
    data.replace(NAN_EQUIVALENTS, inplace=True)
    data = data.apply(_to_numeric)

    # Save Data
    print(f"Saving Data to {file_path}...")
    table = pa.Table.from_pandas(data)
    metadata = table.schema.metadata or {}
    metadata.update({key.encode(): str(value).encode()
                    for key, value in params.items()})
    pq.write_table(table.replace_schema_metadata(metadata), file_path)
    print("Data Saved.")

    return DATASET.get_Xy(data, target)
