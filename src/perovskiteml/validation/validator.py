import numpy as np
import pandas as pd
from neptune import Run
from neptune.types import File
from typing import TYPE_CHECKING, Literal
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import (
    cross_validate,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    LeaveOneOut,
    TimeSeriesSplit
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    make_scorer
)

if TYPE_CHECKING:
    from ..models.base import BaseModelHandler
    


class ValidatorConfig(BaseModel):
    method: Literal[
        "kfold", "stratified_kfold", "shuffle_split", "leave_one_out", "time_series_split"
    ] = "kfold"
    metrics: list[str] = ["rmse"]
    model_config = ConfigDict(extra="allow")


class Validator:
    _cv_registry = {
        "kfold": KFold,
        "shuffle_split": ShuffleSplit,
        "leave_one_out": LeaveOneOut,
    }

    _metrics_registry = {
        "r2": r2_score,
        "r": lambda X, y: np.sqrt(r2_score(X, y)),
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "mae": mean_absolute_error
    }

    def __init__(self, config: dict | ValidatorConfig):
        if isinstance(config, dict):
            config = ValidatorConfig(**config)
        self.config = config
        self.validator = self.create_validator()
        self.scorers = self.create_scorers()
        self._results: dict | None = None

    @classmethod
    def _create_validator(cls, config: ValidatorConfig):
        validator = cls._cv_registry[config.method]
        return validator(**config.model_dump(exclude={"method", "metrics"}))

    @classmethod
    def _create_scorers(cls, config: ValidatorConfig) -> dict:
        scorers = {}
        for name in config.metrics:
            if name not in cls._metrics_registry:
                raise ValueError(f"Unsupported metric: {name}")
            scorers[name] = make_scorer(cls._metrics_registry[name])
        return scorers

    def create_validator(self):
        return self._create_validator(self.config)

    def create_scorers(self) -> dict:
        return self._create_scorers(self.config)

    def cross_validate(self, model: "BaseModelHandler", X: pd.DataFrame, y: pd.Series) -> dict:
        self._results = cross_validate(
            estimator=model.model,
            X=X,
            y=y,
            cv=self.validator,
            scoring=self.scorers,
            return_train_score=True
        )
        return self.results
    
    @property
    def results(self) -> pd.DataFrame | None:
        if self._results:
            df = pd.DataFrame(self._results).drop(
                columns=["fit_time", "score_time"]
            )
            prefix = df.columns.str.split("_").str[0].str.capitalize()
            suffix = df.columns.str.split("_").str[1].str.upper()
            df.columns = prefix + " " + suffix
            return df
        
        return None
            

    @property
    def summary(self) -> pd.DataFrame | None:
        if self._results:
            summary_dict = {
                "Metrics": [],
                "Mean Train": [],
                "Std Train": [],
                "Mean Test": [],
                "Std Test": [],
            }

            for m in self.config.metrics:
                summary_dict["Metrics"].append(m.upper())
                summary_dict["Mean Train"].append(
                    self._results[f"train_{m}"].mean())
                summary_dict["Std Train"].append(
                    self._results[f"train_{m}"].std())
                summary_dict["Mean Test"].append(
                    self._results[f"test_{m}"].mean())
                summary_dict["Std Test"].append(
                    self._results[f"test_{m}"].std())

            return pd.DataFrame(summary_dict)
        
        return None

    def log_metrics(self, run: Run | None = None):
        results = self.results
        summary = self.summary
        
        print("Per-fold metrics:")
        print(results)
        
        print("\nSummary:")
        print(summary)
        
        if run:
            run["per_fold_metrics"].upload(File.as_html(results))
            run["summary"].upload(File.as_html(summary))
            
            for key, value in self._results.items():
                if key not in ["fit_time", "score_time"]:
                    run_key = key.replace("_", "/")
                    run[run_key+"_mean"] = value.mean()
                    run[run_key+"_std"] = value.std()
                
                
                
