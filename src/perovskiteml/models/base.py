import os
import json
import joblib
import numpy as np
from abc import ABC, abstractmethod
from neptune import Run
from optuna import Trial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from pydantic import BaseModel
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from ..validation import Validator


class BaseModelConfig(BaseModel):
    model_type: str


class BaseModelHandler(ABC):
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.callbacks = []

    @abstractmethod
    def create_model(self):
        """Create Model with config params"""
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        """Train model with validation data"""
        pass

    def train(
        self,
        X_train, y_train, X_val, y_val,
        cv: Optional[Validator] = None,
        run: Optional[Run] = None,
        trial: Optional[Trial] = None
    ):
        if cv:
            self.create_model()
            cv.cross_validate(self, X_train, y_train)
            cv.log_metrics(run["validation"] if run else None)
        self.fit(X_train, y_train, X_val, y_val)
        self.log_metrics(X_train, y_train, prefix="train", run=run)
        self.log_metrics(X_val, y_val, prefix="val", run=run)
        self.log_additional_info(run)

    def predict(self, X):
        """Generate Predictions"""
        return self.model.predict(X)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path = path / self.config.model_type
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.joblib"
        config_path = path / "config.json"
        joblib.dump(self.model, model_path, compress=3)
        with open(config_path, "w") as file:
            json.dump(self.config.model_dump(), file, indent=4)

    @classmethod
    def load(cls, path: Path | str):
        path = Path(path)
        model_path = path / "model.joblib"
        config_path = path / "config.json"
        model = joblib.load(model_path)
        with open(config_path, "r") as file:
            config = json.load(file)
        instance = cls(config)
        instance.model = model
        return instance

    def init_callbacks(
        self, run: Optional[Run] = None, trial: Optional[Trial] = None
    ):
        """initialize model callbacks"""
        pass

    def log_additional_info(self, run: Run = None):
        """Log model-specific information to Neptune"""
        if run:
            with TemporaryDirectory() as tmp_file:
                model_file = os.path.join(
                    tmp_file,
                    f"{self.config.model_type}\\model.joblib"
                )
                self.save(tmp_file)
                print(f"Uploading Model from: {model_file}")
                run["model"].upload(model_file, wait=True)
        pass

    def calculate_metrics(self, X, y):
        predicted = self.predict(X)
        r2 = r2_score(predicted, y)
        mae = mean_absolute_error(predicted, y)
        mse = mean_squared_error(predicted, y)
        r = np.sqrt(r2)
        rmse = np.sqrt(mse)
        return r2, r, mae, mse, rmse

    @staticmethod
    def _log_metrics_to_stdout(
        r2: float, r: float, mae: float, mse: float, rmse: float, prefix: str
    ):
        print(f"\nR2 on {prefix} Set:", r2)
        print(f"R value on {prefix} Set:", r)
        print(f"MAE on {prefix} Set:", mae)
        print(f"MSE on {prefix} Set:", mse)
        print(f"RMSE on {prefix} Set:", rmse)

    @staticmethod
    def _log_metrics_to_neptune(
        r2: float, r: float, mae: float, mse: float, rmse: float, prefix: str, run: Run
    ):
        run[f"metrics/{prefix}/r2"] = r2
        run[f"metrics/{prefix}/mae"] = mae
        run[f"metrics/{prefix}/mse"] = mse
        run[f"metrics/{prefix}/r"] = r
        run[f"metrics/{prefix}/rmse"] = rmse

    def log_metrics(self, X, y, prefix="val", run: Run = None):
        """Log shared metrics"""
        r2, r, mae, mse, rmse = self.calculate_metrics(X, y)
        self._log_metrics_to_stdout(r2, r, mae, mse, rmse, prefix)

        if run:
            self._log_metrics_to_neptune(r2, r, mae, mse, rmse, prefix, run)


class ModelFactory:
    _models = {}
    _configs = {}

    @classmethod
    def register_model(cls, name: str):
        def decorator(model_cls):
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def register_config(cls, name: str):
        def decorator(config_cls):
            cls._configs[name] = config_cls
            return config_cls
        return decorator

    @classmethod
    def create(cls, config: dict | BaseModelConfig) -> BaseModelHandler:
        if isinstance(config, dict):
            config = cls._validate_dict_config(config)

        if config.model_type not in cls._models:
            raise ValueError(
                f"Unregistered model type: {config.model_type}. "
                f"Registered models include: {cls._models.keys()}"
            )

        return cls._models[config.model_type](config)

    @classmethod
    def _validate_dict_config(cls, config: dict) -> BaseModelConfig:
        model_type = config.get("model_type")

        if model_type not in cls._configs:
            raise ValueError(
                f"Unregistered model type: {model_type}. "
                f"Registered models include: {cls._models.keys()}"
            )

        return cls._configs[model_type](**config)
