import json
import pandas as pd

from typing import Union
from pathlib import Path
from pydantic import BaseModel
from .base import (
    BaseDataset,
    DatasetMetadata,
    DataSavingError,
    DataLoadingError
)

# --------------------------
# Folder Structure Config
# --------------------------


class DatabaseConfig(BaseModel):
    """
    Configuration for the file structure of the database.
    """
    data_file: str = "perovskite_database.csv"
    reference_file: str = "pdp_units_data.xlsx"
    section_key_file: str = "section_keys.json"
    nan_equivalents_file: str = "nan_equivalents.json"

# --------------------------
# Perovskite Database
# --------------------------


class PerovskiteDatabase(BaseDataset):
    """
    Perovskite database class.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: DatasetMetadata,
        reference: pd.DataFrame,
        section_keys: dict,
        nan_equivalents: dict,
        config: DatabaseConfig = DatabaseConfig(),
    ):
        self._reference = reference
        self._section_keys = section_keys
        self._nan_equivalents = nan_equivalents
        self.config = config
        super().__init__(data, metadata)

    @classmethod
    def load(
        cls,
        folder_path: Union[Path, str],
        config: DatabaseConfig = DatabaseConfig(),
    ) -> 'PerovskiteDatabase':

        try:
            folder_path = Path(folder_path)
            if not folder_path.is_dir():
                raise FileNotFoundError(
                    f"Directory {folder_path} does not exist.")

            data_path = folder_path / config.data_file
            data = pd.read_csv(data_path, low_memory=False)

            reference_path = folder_path / config.reference_file
            reference = pd.read_excel(reference_path, sheet_name=None)

            section_key_path = folder_path / config.section_key_file
            section_keys = cls._load_json(section_key_path)

            nan_equivalents_path = folder_path / config.nan_equivalents_file
            nan_equivalents = cls._load_json(nan_equivalents_path)
            
            data.replace(nan_equivalents, inplace=True)
            metadata = cls._get_dataframe_metadata(data, data_path)

        except Exception as e:
            raise DataLoadingError(
                f"Error loading dataset from {folder_path}: {e}"
            ) from e

        return cls(
            data=data,
            metadata=metadata,
            reference=reference,
            section_keys=section_keys,
            nan_equivalents=nan_equivalents,
            config=config
        )

    def save(self, folder_path: Path) -> None:
        # TODO: This functionality is only necessary once data cleansing is automated.
        raise NotImplementedError

    # --------------------------
    # Data Operations
    # --------------------------

    @property
    def reference(self) -> pd.DataFrame:
        return self._reference

    @property
    def section_keys(self) -> dict:
        return self._section_keys

    @property
    def nan_equivalents(self) -> dict:
        return self._nan_equivalents
