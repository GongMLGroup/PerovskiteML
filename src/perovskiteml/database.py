import os
import copy
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .fileutils import PROJECT_DIR, DATA_DIR, hash_params
from .preprocess import preprocess_data
from .logger import data_logger, setup_logger


class PerovskiteData():
    """Stores the unprocessed perovskite data.

    Attributes:
        data (dataframe): The unprocessed data.
        ref (dataframe): The reference data for features.
            - Field
            - Type
            - Default
            - Unit
            - Pattern
            - Implemented
            - Description
            - Concerns
        database_file (str): The name of the database file.
        ref_file (str): The name of the reference data file.
        nan_equivalents (dict): Equivalent values for NaN in the dataset.
        section_keys (dict): Dictionary of section names and their corresponding shorthand.
        X (dataframe): The preprocessed and masked data.
            Stored after preprocess() is called.
        y (series): The masked target
            Stored after preprocess() is called.

    Raises:
        ValueError: If both ref and ref_file are None.
        ValueError: If both data and database_file are None.

    """

    def __init__(self, ref=None, data=None, ref_file=None, database_file=None, nan_equivalents={}, section_keys={}):
        if (ref is None) and (ref_file is None):
            ValueError("ref or ref_file must be provided")
        if (data is None) and (database_file is None):
            ValueError("data or database_file must be provided")
        self.data = data
        self.ref = ref
        self.database_file = database_file
        self.ref_file = ref_file
        self.nan_equivalents = nan_equivalents
        self.section_keys = section_keys
        self.X = None
        self.y = None

    def load_data(self, verbosity: int = 0):
        setup_logger(verbosity)
        """Loads the reference and database data."""
        if self.data is None:
            data_logger.info("Loading Perovskite Data.")
            if not os.path.isabs(self.database_file):
                self.database_file = os.path.join(DATA_DIR, self.database_file)
            self.data = pd.read_csv(self.database_file, low_memory=False)
            self.data.replace(self.nan_equivalents, inplace=True)

        if self.ref is None:
            data_logger.info("Loading Reference Data.")
            if not os.path.isabs(self.ref_file):
                self.ref_file = os.path.join(DATA_DIR, self.ref_file)
            self.ref = pd.read_excel(self.ref_file, sheet_name=None)

        data_logger.info("Data Loaded.")

    def get_Xy(self, data, target):
        """Returns a masked version of the data and target series.

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

    def set_Xy(self, data, target):
        """Calls `getXy(data, target)` and stores the output."""
        X, y = self.get_Xy(data, target)
        self.X = X
        self.y = y
        return X, y

    def preprocess(self, target, threshold, depth, exclude_sections=[], exclude_cols=[], save: bool = True, verbosity: int = 0):
        """Generates a preprocessed version of the dataset.
        
        If an unseen set of hyperparameters is used to generate the preprocessed dataset, it is saved for future use. Otherwise, the previously generated file is loaded and returned instead.
        
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
            save (bool, optional): Whether to save the preprocessed data.
                Defaults to True.
            verbosity (int, optional): Verbosity level.
                Defaults to 0.

        Returns:
            dataframe: The preprocessed data.
            series: The target data.

        """
        self.load_data(verbosity=verbosity)

        params = {
            'target': target,
            'threshold': threshold,
            'depth': depth,
            'exclude_sections': exclude_sections,
            'exclude_cols': exclude_cols,
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
            data_logger.info("File already exists.")
            data_logger.debug(f"Found at: {file_path}")
            table = pq.read_table(file_path)
            data_logger.info("Preprocessed data Loaded.")
            return self.set_Xy(table.to_pandas(), target)

        # Remove excluded keys
        sections = copy.copy(self.section_keys)
        for key in exclude_sections:
            del sections[key]

        data_logger.info("File does not exist.")

        data = preprocess_data(
            self.data,
            self.ref,
            threshold,
            depth,
            sections=sections,
            exclude_cols=exclude_cols,
            nan_equivalents=self.nan_equivalents,
            verbosity=verbosity
        )

        # Save Data
        if save:
            data_logger.info("Saving Data.")
            data_logger.debug(f"Saving to: {file_path}")
            table = pa.Table.from_pandas(data)
            metadata = table.schema.metadata or {}
            metadata.update({key.encode(): str(value).encode()
                            for key, value in params.items()})
            pq.write_table(table.replace_schema_metadata(metadata), file_path)
            data_logger.info("Data Saved.")

        return self.set_Xy(data, target)
    

with open(os.path.join(PROJECT_DIR,"data/section_keys.json"), "r") as file:
    SECTION_KEYS = json.load(file)
    """dict: Dictionary of section names and their corresponding shorthand.

    The shorhand is used as a prefix for feature names.
    """
with open(os.path.join(PROJECT_DIR,"data/nan_equivalents.json"), "r") as file:
    NAN_EQUIVALENTS = json.load(file)
    """dict: Keys are equivalent to `missing` or `nan` values in the dataset."""

DATABASE = PerovskiteData(
    ref_file="pdp_units_data.xlsx",
    database_file="Perovskite_database.csv",
    nan_equivalents=NAN_EQUIVALENTS,
    section_keys=SECTION_KEYS
)
"""An instance of the Perovskite Dataset."""
