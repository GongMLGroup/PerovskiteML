from sklearn.neighbors import KNeighborsRegressor
from pydantic import Field, ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("knn")
class KNeighborsConfig(BaseModelConfig):
    model_type: Literal["knn"] = "knn"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    n_jobs: int = -1
    model_config = ConfigDict(extra="allow")
        
@ModelFactory.register_model("knn")
class KNeighborsHandler(BaseModelHandler):
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model = KNeighborsRegressor(
            **self.config.model_dump(exclude={"model_type"})
        )
        self.model.fit(
            X_train, y_train
        )
    