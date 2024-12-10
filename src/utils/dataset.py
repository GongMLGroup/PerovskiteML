import os
import pandas as pd

from .fileutils import DATA_DIR

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

    Raises:
        ValueError: If both ref and ref_file are None.
        ValueError: If both data and database_file are None.

    """
    def __init__(self, ref=None, data=None, ref_file=None, database_file=None, nan_equivalents={}, section_keys={}):
        if (ref is None) and (ref_file is None):
            ValueError("ref or ref_file must be provided")
        if (data is None) and (database_file is None):
            ValueError("data or database_file must be provided")
        self.ref = ref
        self.data = data
        self.ref_file = ref_file
        self.database_file = database_file
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
