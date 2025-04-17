import hashlib
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Union


# --------------------------
# Custom Exceptions
# --------------------------

class DataLoadingError(Exception):
    """Base exception for errors during data loading."""
    pass


class InvalidFileFormatError(DataLoadingError):
    """Raised when file format is unsupported"""
    pass


class DataValidationError(Exception):
    """Raised when dataset fails validation checks"""
    pass


class DataSavingError(Exception):
    """Base exception for data saving failures"""
    pass

# --------------------------
# Metadata Model
# --------------------------


class DatasetMetadata(BaseModel):
    creation_date: datetime = Field(default_factory=datetime.now)
    data_hash: str = ""
    source_path: Path = Path("")
    feature_count: int = 0
    sample_count: int = 0
    target_feature: str = ""
    processing_history: list[str] = []
    tags: list[str] = []
    
# --------------------------
# Dataset Config
# --------------------------


class DatasetConfig(BaseModel):
    target_feature: str | list[str]
    model_config = ConfigDict(extra="allow")
    

# --------------------------
# Base Dataset Class
# --------------------------


class BaseDataset:
    SUPPORTED_FORMATS = {'.csv', '.parquet'}

    def __init__(self, data: pd.DataFrame, metadata: DatasetMetadata = DatasetMetadata()):
        self._data = data
        self._metadata = metadata
        self._update_metadata()
        # self._validate()

    @classmethod
    def load(cls, path: Union[Path, str]) -> 'BaseDataset':
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")

            suffix = path.suffix.lower()

            match suffix:
                case '.csv':
                    data, metadata = cls._load_csv(path)
                case '.parquet':
                    data, metadata = cls._load_parquet(path)
                case _:
                    raise InvalidFileFormatError(
                        f"Unsupported file format: {suffix}. "
                        f"Supported formats: {cls.SUPPORTED_FORMATS}"
                    )
            return cls(data, metadata)

        except Exception as e:
            raise DataLoadingError(
                f"Error loading dataset from {path}: {e}"
            ) from e

    def save(self, path: Union[Path, str], overwrite: bool = False) -> None:
        try:
            path = Path(path)
            if path.exists() and not overwrite:
                raise FileExistsError(f"File alread exists: {path}")

            suffix = path.suffix.lower()

            match suffix:
                case '.csv':
                    self._save_csv(path)
                case '.parquet':
                    self._save_parquet(path)
                case _:
                    raise DataSavingError(
                        f"Unsupported file format: {suffix}. "
                        f"Supported formats: {self.SUPPORTED_FORMATS}"
                    )

            self._update_metadata(path)

        except Exception as e:
            raise DataSavingError(
                f"Error saving dataset to {path}: {e}"
            ) from e

    def _validate(self) -> None:
        if self.metadata.sample_count != len(self.data):
            raise DataValidationError(
                f"Metadata sample count ({self.metadata.sample_count}) "
                f"does not match data length ({len(self.data)})"
            )

        current_hash = self.calculate_hash()
        if current_hash != self.metadata.data_hash:
            raise DataValidationError(
                "Dataset hash mismatch. Data may be corrupted or modified."
            )

    def calculate_hash(self) -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(self.data).values.tobytes()
        ).hexdigest()

    def _update_metadata(self, path: Path = "") -> None:
        if path:
            self.metadata.source_path = path
        self.metadata.sample_count, self.metadata.feature_count = self.data.shape
        self.metadata.data_hash = self.calculate_hash()
        self.metadata.creation_date = datetime.now()

    # --------------------------
    # Format Implementations
    # --------------------------

    @classmethod
    def _load_csv(cls, path: Path) -> 'BaseDataset':
        data = pd.read_csv(path, low_memory=False)
        metadata = cls._get_dataframe_metadata(data, path)
        return data, metadata

    @classmethod
    def _load_parquet(cls, path: Path) -> 'BaseDataset':
        table = pq.read_table(path)
        data = table.to_pandas()
        metadata = cls._get_parquet_metadata(table)
        return data, metadata

    def _save_csv(self, path: Path) -> None:
        self.data.to_csv(path, index=False)

    def _save_parquet(self, path: Path) -> None:
        table = pa.Table.from_pandas(self.data)
        self._data = table.to_pandas()
        self._update_metadata(path)
        metadata = table.schema.metadata or {}
        metadata.update(
            {b'data_metadata': self.metadata.model_dump_json().encode('utf-8')})
        schema = table.schema.with_metadata(metadata)
        table = table.cast(schema)
        pq.write_table(table, path)

    # --------------------------
    # Metadata Handling
    # --------------------------

    @staticmethod
    def _get_dataframe_metadata(data: pd.DataFrame, source_path: Path) -> DatasetMetadata:
        return DatasetMetadata(
            creation_date=datetime.now(),
            data_hash=hashlib.sha256(
                pd.util.hash_pandas_object(data).values.tobytes()
            ).hexdigest(),
            source_path=source_path,
            feature_count=len(data.columns),
            sample_count=len(data)
        )

    @staticmethod
    def _get_parquet_metadata(table: pa.Table) -> DatasetMetadata:
        metadata = table.schema.metadata[b'data_metadata']
        return DatasetMetadata(**json.loads(metadata))

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)

    # --------------------------
    # Data Operations
    # --------------------------

    def split_target(
        self,
        target_column: str,
        drop_target: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        target = self.data[target_column]
        if drop_target:
            data = self.data.drop(columns=[target_column])
        else:
            data = self.data
            
        mask = target.notna()
        return data[mask], target[mask]

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata
