from sklearn.neighbors import KNeighborsRegressor
from pydantic import ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("knn")
class KNeighborsConfig(BaseModelConfig):
    model_type: Literal["knn"] = "knn"
    n_jobs: int = -1
    model_config = ConfigDict(extra="allow")
        
@ModelFactory.register_model("knn")
class KNeighborsHandler(BaseModelHandler):
    def __init__(self, config: KNeighborsConfig):
        super().__init__(config)
        self.config = config
        self.create_model()
        
    def create_model(self):
        self.model = KNeighborsRegressor(
            **self.config.model_dump(exclude={"model_type"})
        )
    
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.create_model()
        self.model.fit(
            X_train, y_train
        )
    