import os
import copy
import dotenv
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# This File acts a pseudo package, allowing useful scripts to be easily imported
from .data_expansion import *
from .data_reduction import *

dotenv.load_dotenv("../../.env")
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

SECTION_KEYS = {
    "Reference information": "Ref",
    "Cell definition": "Cell",
    "Module definition": "Module",
    "Substrate": "Substrate",
    "Electron transport layer": "ETL",
    "The perovskite": "Perovskite",
    "Perovskite deposition": "Perovskite_Dep",
    "Hole transport layer": "HTL",
    "Back contact": "Backcontact",
    "Additional layers": "Add_lay",
    "Encapsulation": "Encapsulation",
    "JV data": "JV",
    "Stabilised efficiency": "Stabilised",
    "Quantum efficiency": "EQE",
    "Stability": "Stability",
    "Outdoor testing": "Outdoor",
}
"""dict: Dictionary of section names and their corresponding shorthand.

The shorhand is used as a prefix for feature names.
"""

NAN_EQUIVALENTS = {
    'Unknown': None,
    'unknown': None,
    'None': None,
    'none': None,
    'NULL': None,
    'Null': None,
    'null': None,
    'NAN': None,
    'NaN': None,
    'Nan': None,
    'nan': None,
    'NAN; NAN': None,
    'NaN; NaN': None,
    'Nan; Nan': None,
    'nan; nan': None
}
"""dict: Keys are equivalent to `missing` or `nan` values in the dataset."""


def _search_parents(path, depth=2):
    """Given a relative path search its parents until it's found from the current working directory.

    This allows a folder name to be input and the relative path from the current working directory to be found.

    Args:
        path (str): The child directory.
        depth (int, optional): The maximum number of steps out of the child directory. Defaults to 2.

    Returns:
        str: A vaild relative path to the child directory. 

    Raises:
        FileNotFoundError: If the path is not found.

    """
    for x in range(depth):
        if os.path.exists(path):
            return path
        path = os.path.join('../', path)
    if os.path.exists(path):
        return path
    return FileNotFoundError(f"Path not found: {path}")


class PerovskiteData():
    """Stores the unprocessed perovskite data.
    
    Attributes:
        ref (dataframe): The reference data for features
            - Field
            - Type
            - Default
            - Unit
            - Pattern
            - Implemented
            - Description
            - Concerns
        data (dataframe): The unprocessed data.
        ref_file (str): The name of the reference data file.
        database_file (str): The name of the database file.

    """
    ref = None
    data = None
    ref_file = 'pdp_units_data.xlsx'
    database_file = 'Perovskite_database.csv'

    def load_data(self):
        """Loads the reference and database data."""
        if self.data is None:
            print("Loading Perovskite Data...")
            database_path = os.path.join('data/', self.database_file)
            database_path = _search_parents(database_path)
            self.data = pd.read_csv(database_path, low_memory=False)
            self.data.replace(NAN_EQUIVALENTS, inplace=True)

        if self.ref is None:
            print("Loading Reference Data...")
            ref_path = os.path.join('data/', self.ref_file)
            ref_path = _search_parents(ref_path)
            self.ref = pd.read_excel(ref_path, sheet_name=None)

        print("Data Initialized.")

    def get_Xy(self, data, target):
        """ Returns a masked version of the data and target series.
        
        Masks data against the target series excluding NaN target values.
        
        Args:
            data (dataframe): The perovskite data.
            target (str): The name of the target feature.
            
        Returns:
            dataframe: The masked data.
            series: The masked target.
            
        """
        # Mask data against target. Target values cannot be NaN
        mask = self.data[target].notna()
        X = data[mask]
        y = self.data[mask][target]
        return X, y


DATASET = PerovskiteData()
"""An instance of the Perovskite Dataset."""


def _column_selector(pat, nonpat):
    """[DEPRICATED]"""
    patterned = [col for col in pat]
    categorical = []
    numerical = []
    for col in nonpat:
        col_types = nonpat[col].apply(type)
        if np.any((col_types == bool) | (col_types == object) | (col_types == str)):
            categorical.append(col)
        else:
            numerical.append(col)
    return {
        'patterned': patterned,
        'categorical': categorical,
        'numerical': numerical
    }


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
    file_path = _search_parents('data/preprocessed')
    file_path = os.path.join(file_path, file_name)

    # Check if file exists
    if os.path.exists(file_path):
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


def display_scores(scores):
    """Print formatted Mean and Std"""
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(
        scores, np.mean(scores), np.std(scores)))
