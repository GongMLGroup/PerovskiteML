import numpy as np
from pandas import DataFrame, Series
from typing import Literal
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from pydantic import BaseModel, ConfigDict, Field

class StepConfig(BaseModel):
    type: Literal["impute", "encode", "scale", "other"] = "other"
    method: str = "passthrough"
    model_config = ConfigDict(extra="allow")

class PreprocessorConfig(BaseModel):
    numerical: list[StepConfig] = Field(default_factory=list[StepConfig])
    categorical: list[StepConfig] = Field(default_factory=list[StepConfig])

class Preprocessor:
    _step_registry = {
        "impute": {
            "simple": SimpleImputer,
            "knn": KNNImputer
        },
        "encode": {
            "onehot": OneHotEncoder,
            "ordinal": OrdinalEncoder
        },
        "scale": {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler
        },
        "other": {
            "passthrough": lambda **kwargs: "passthrough",
            "drop": lambda **kwargs: "drop"
        }
    }
    
    numerical_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=[bool, object])
    
    def __init__(self, config: dict | PreprocessorConfig):
        if isinstance(config, dict):
            config = PreprocessorConfig(**config)
        self.config = config
        self.build_pipeline()
    
    @classmethod
    def _build_pipeline(cls, steps: list[StepConfig]) -> Pipeline:
        name_step_pairs: list[tuple] = []
        for step in steps:
            config = step.model_dump(exclude=["type", "method"])
            pair = (
                step.type,
                cls._step_registry[step.type][step.method](**config)
            )
            name_step_pairs.append(pair)
        return Pipeline(name_step_pairs)
        
    def build_pipeline(self):
        self.pipeline = ColumnTransformer([
            (
                "numerical",
                self._build_pipeline(self.config.numerical),
                self.numerical_selector
            ),
            (
                "categorical",
                self._build_pipeline(self.config.categorical),
                self.categorical_selector
            )
        ])
        
    def preprocess(self, X: DataFrame , y: Series) -> tuple[DataFrame, Series]:
        data_transformed = DataFrame(self.pipeline.fit_transform(X, y))
        numerical_columns = list(X.select_dtypes(np.number).columns)
        categorical_columns = list(X.select_dtypes([bool, object]).columns)
        data_transformed.columns = numerical_columns + categorical_columns
        return data_transformed
    
        