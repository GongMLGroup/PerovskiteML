import catboost as cb
from optuna.integration.catboost import CatBoostPruningCallback
from pydantic import ConfigDict
from typing import Literal
from .base import BaseModelConfig, BaseModelHandler, ModelFactory


@ModelFactory.register_config("catboost")
class CatboostConfig(BaseModelConfig):
    model_type: Literal["catboost"] = "catboost"
    eval_metric: str = "RMSE"
    verbose: bool = True
    random_state: int = 42
    allow_writing_files: bool = False
    model_config = ConfigDict(extra="allow")
    

@ModelFactory.register_model("catboost")
class CatboostHandler(BaseModelHandler):
    def __init__(self, config: CatboostConfig):
        super().__init__(config)
        self.config = config
        self.model = cb.CatBoostRegressor(
            **self.config.model_dump(exclude={"model_type"}),
        )
        
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=self.config.verbose,
            callbacks=self.callbacks
        )
        
    def init_callbacks(self, run = None, trial = None) -> None:            
        if trial:
            self.callbacks.append(
                CatBoostPruningCallback(trial, self.config.eval_metric, 0)
            )
    