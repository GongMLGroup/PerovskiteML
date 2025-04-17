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
    def __init__(self, config: RandomForestConfig):
        super().__init__(config)
        self.config = config
        self.create_model()
        
    def create_model(self):
        self.model = RandomForestRegressor(
            **self.config.model_dump(exclude="model_type")
        )
        
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.create_model()
        self.model.fit(
            X_train, y_train
        )
    