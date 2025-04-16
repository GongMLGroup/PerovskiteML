from sklearn.ensemble import HistGradientBoostingRegressor
from pydantic import ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("hist_gradient_boost")
class HistGradientBoostingConfig(BaseModelConfig):
    model_type: Literal["hist_gradient_boost"] = "hist_gradient_boost"
    random_state: int = 42
    verbose: bool = False
    model_config = ConfigDict(extra="allow")


@ModelFactory.register_model("hist_gradient_boost")
class HistGradientBoostingHandler(BaseModelHandler):
    def __init__(self, config: HistGradientBoostingConfig):
        super().__init__(config)
        self.config = config
        self.model = HistGradientBoostingRegressor(
            **self.config.model_dump(exclude="model_type")
        )
        
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model.fit(
            X_train, y_train
        )