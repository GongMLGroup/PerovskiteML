import os
import copy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .fileutils import DATA_DIR, hash_params
from .preprocess import preprocess_data

class PerovskiteData():
    """Stores the unprocessed perovskite data.

    Attributes:
        data (dataframe): The unprocessed data.
        ref (dataframe): The reference data for features
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

    def load_data(self):
        """Loads the reference and database data."""
        if self.data is None:
            print("Loading Perovskite Data...")
            if not os.path.isabs(self.database_file):
                self.database_file = os.path.join(DATA_DIR, self.database_file)
            self.data = pd.read_csv(self.database_file, low_memory=False)
            self.data.replace(self.nan_equivalents, inplace=True)

        if self.ref is None:
            print("Loading Reference Data...")
            if not os.path.isabs(self.ref_file):
                self.ref_file = os.path.join(DATA_DIR, self.ref_file)
            self.ref = pd.read_excel(self.ref_file, sheet_name=None)

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

    def set_Xy(self, data, target):
        X, y = self.get_Xy(data, target)
        self.X = X
        self.y = y
        return X, y
    
    def preprocess(self, target, threshold, depth, exclude_sections=[], exclude_cols=[], verbose: bool = True):
        self.load_data()

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
            print(f'File already exists: {file_path}')
            print('Loading Data...')
            table = pq.read_table(file_path)
            return self.set_Xy(table.to_pandas(), target)
        
        # Remove excluded keys
        sections = copy.copy(self.section_keys)
        for key in exclude_sections:
            del sections[key]

        print(f"File does not exist, preprocessing and saving to {file_path}")
        print("Preprocessing Data...")
        print(f"Threshold: {threshold}, Depth: {depth}")
        
        data = preprocess_data(
            self.data,
            self.ref,
            threshold,
            depth,
            sections=sections,
            exclude_cols=exclude_cols,
            nan_equivalents=self.nan_equivalents,
            verbose=verbose
        )
        
        # Save Data
        print(f"Saving Data to {file_path}...")
        table = pa.Table.from_pandas(data)
        metadata = table.schema.metadata or {}
        metadata.update({key.encode(): str(value).encode()
                        for key, value in params.items()})
        pq.write_table(table.replace_schema_metadata(metadata), file_path)
        print("Data Saved.")
        
        return self.set_Xy(data, target)

