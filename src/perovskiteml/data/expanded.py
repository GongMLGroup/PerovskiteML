import pandas as pd

from copy import deepcopy
from pathlib import Path
from pydantic import BaseModel, computed_field
from typing import Tuple

from .base import (
    BaseDataset,
    DataLoadingError,
    DataValidationError,
    DatasetMetadata
)
from .perovskite import PerovskiteDatabase

from ..preprocessing.expansion import expand_sort
from ..preprocessing.preprocess import _to_numeric


# --------------------------
# Folder Structure Config
# --------------------------


class ExpansionConfig(BaseModel):
    """
    Configuration for the expansion of the dataset.
    """
    target_feature: str = ""
    force_recompute: bool = False
    cache_dir: Path = Path("data/expanded")
    source_dir: Path = Path("data/clean")
    cache_file: Path = Path("dataset.parquet")
    features_file: Path = Path("features.json")
    section_keys_file: Path = Path("section_keys.json")

    @computed_field
    @property
    def target_cache_dir(self) -> Path:
        return self.cache_dir / self.target_feature

    @computed_field
    @property
    def target_cache_path(self) -> Path:
        return self.target_cache_dir / self.cache_file

    @computed_field
    @property
    def target_features_path(self) -> Path:
        return self.target_cache_dir / self.features_file

    @computed_field
    @property
    def section_keys_path(self) -> Path:
        return self.source_dir / self.section_keys_file


# --------------------------
# Expanded Dataset
# --------------------------


class ExpandedDataset(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        metadata: DatasetMetadata,
        features: list,
        reference: dict,
        config: ExpansionConfig = ExpansionConfig()
    ):
        self._all_features = features
        self._features = deepcopy(self._all_features)
        self._reference = reference
        self.config = config
        super().__init__(data, metadata)

    @classmethod
    def cache_or_compute(
        cls,
        config: ExpansionConfig = ExpansionConfig(),
        **kwargs
    ) -> 'ExpandedDataset':
        
        if isinstance(config, dict):
            config = ExpansionConfig(**config)
        if kwargs:
            config = ExpansionConfig(**kwargs)

        if not config.target_feature:
            raise DataValidationError("Target feature cannot be empty.")

        # Try to load the cached dataset
        if not config.target_cache_dir.exists():
            config.target_cache_dir.mkdir(parents=True, exist_ok=True)
        elif not config.force_recompute and cls._has_dataset(config):
            try:
                return cls.load(config)
            except (DataLoadingError, DataValidationError):
                pass

        # Compute the data expansion if needed
        source_dataset = PerovskiteDatabase.load(config.source_dir)
        expanded_data, features = cls._perform_expansion(
            config.target_feature, source_dataset)

        # Save the expanded dataset
        metadata = cls._get_dataframe_metadata(
            expanded_data, config.target_cache_path)
        metadata.target_feature = config.target_feature
        instance = cls(
            data=expanded_data,
            metadata=metadata,
            features=features,
            reference=source_dataset.section_keys,
            config=config
        )
        instance.save(overwrite=True)
        return instance

    @classmethod
    def load(
        cls,
        config: ExpansionConfig = ExpansionConfig(),
        **kwargs
    ) -> 'ExpandedDataset':

        if isinstance(config, dict):
            config = ExpansionConfig(**config)
        if kwargs:
            config = ExpansionConfig(**kwargs)

        data, metadata = cls._load_parquet(config.target_cache_path)
        features = cls._load_json(config.target_features_path)
        reference = cls._load_json(config.section_keys_path)

        return cls(data, metadata, features, reference, config)

    def save(self, overwrite: bool = False) -> None:
        super().save(path=self.config.target_cache_path, overwrite=overwrite)
        self._save_json(self.config.target_features_path, self._all_features)

    def reset_features(self) -> None:
        self._features = deepcopy(self._all_features)
        self.metadata.processing_history = []
        return

    @staticmethod
    def _has_dataset(config: ExpansionConfig) -> bool:
        if not config.target_cache_path.exists():
            return False
        if not config.target_features_path.exists():
            return False
        return True

    # --------------------------
    # Expansion Handling
    # --------------------------

    @staticmethod
    def _perform_expansion(
        target_feature: str,
        source_dataset: PerovskiteDatabase
    ) -> pd.DataFrame:
        data, _ = source_dataset.split_target(
            target_feature, drop_target=False)
        expanded_data, features = expand_sort(
            data, source_dataset.reference['Full Table'])
        expanded_data = expanded_data.apply(_to_numeric)
        return expanded_data, features

    # --------------------------
    # Data Operations
    # --------------------------

    @staticmethod
    def _collect_features(features: list) -> list:
        feature_array = []
        for feature_set in features:
            feature_array.extend(feature_set['children'])
        return feature_array

    def _section_features(self, sections: list):
        return [feature for section in sections for feature in self.reference[section]]

    def remove_features(self, remove_features: list) -> None:
        _features = []
        for feature in self._features:
            if feature['parent'] not in remove_features:
                _features.append(feature)
        self._features = _features
        return

    def remove_sections(self, remove_sections: list) -> None:
        self.remove_features(self._section_features(remove_sections))
        return

    def remove(self, remove_features: list, remove_sections: list) -> None:
        remove_list = self._section_features(remove_sections)
        remove_list.extend(remove_features)
        self.remove_features(remove_list)
        return
    
    def split_target(self, drop_target: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        data = self.data[self.features]
        if drop_target and (self.config.target_feature in data.columns):
            data = data.drop(columns=[self.config.target_feature])
        return data, self.data[self.config.target_feature]

    @property
    def all_features(self) -> dict:
        return self._collect_features(self._all_features)

    @property
    def reference(self) -> dict:
        return self._reference

    @property
    def features(self) -> dict:
        return self._collect_features(self._features)
