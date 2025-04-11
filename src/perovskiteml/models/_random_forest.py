from sklearn.ensemble import RandomForestRegressor
from pydantic import ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("random_forest")
class RandomForestConfig(BaseModelConfig):
    model_type: Literal["random_forest"]
    n_jobs: int = -1
    verbose: bool = False
    model_config = ConfigDict(extra="allow")


@ModelFactory.register_model("random_forest")
class RandomForestHandler(BaseModelHandler):
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model = RandomForestRegressor(
            **self.config.model_dump(exclude="model_type")
        )
        self.model.fit(
            X_train, y_train
        )
    