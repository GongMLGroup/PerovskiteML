from sklearn.ensemble import HistGradientBoostingRegressor
from pydantic import Field, ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("hist_gradient_boost")
class HistGradientBoostingConfig(BaseModelConfig):
    model_type: Literal["hist_gradient_boost"] = "hist_gradient_boost"
    learning_rate: float = Field(0.1, ge=0)
    random_state: int = 42
    max_depth: int | None = None
    max_iter: int = Field(100, ge=1)
    verbose: bool = False
    model_config = ConfigDict(extra="allow")


@ModelFactory.register_model("hist_gradient_boost")
class HistGradientBoostingHandler(BaseModelHandler):
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model = HistGradientBoostingRegressor(
            **self.config.model_dump(exclude="model_type")
        )
        self.model.fit(
            X_train, y_train
        )