import xgboost as xgb
from pydantic import Field
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("xgboost")
class XGBoostConfig(BaseModelConfig):
    model_type: Literal["xgboost"] = "xgboost"
    n_estimators: int = Field(100, ge=1)
    early_stopping_rounds: int | None = None
    eta: float = Field(0.3, ge=0)
    max_depth: int = Field(6, ge=1)
    random_state: int = 42
    

@ModelFactory.register_model("xgboost")
class XGBoostHandler(BaseModelHandler):
    def fit(self, X_train, y_train, X_val, y_val):
        self.model = xgb.XGBRegressor(**self.config.model_dump(exclude="model_type"))
        self.model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)]
        )
    
    def log_additional_info(self, run):
        run["feature_importance"] = list(self.model.feature_importances_)