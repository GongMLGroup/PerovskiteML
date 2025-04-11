import xgboost as xgb
from neptune.integrations.xgboost import NeptuneCallback
from optuna.integration.xgboost import XGBoostPruningCallback
from pydantic import ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("xgboost")
class XGBoostConfig(BaseModelConfig):
    model_type: Literal["xgboost"] = "xgboost"
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True
    eval_metric: str = "rmse"
    model_config = ConfigDict(extra="allow")


@ModelFactory.register_model("xgboost")
class XGBoostHandler(BaseModelHandler):
    def __init__(self, config: XGBoostConfig):
        super().__init__(config)
        self.config = config
        
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model = xgb.XGBRegressor(
            **self.config.model_dump(exclude={"model_type", "verbose"}),
            callbacks=self.callbacks
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=self.config.verbose
        )

    def init_callbacks(self, run=None, trial=None) -> None:
        if run:
            self.callbacks.append(
                NeptuneCallback(run=run, log_model=False, log_importance=False)
            )
        
        if trial:
            self.callbacks.append(
                XGBoostPruningCallback(trial, "validation_1-" + self.config.eval_metric)
            )
        
