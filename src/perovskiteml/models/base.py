import json
import joblib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr


class BaseModelConfig(BaseModel):
    model_type: str


class BaseModelHandler(ABC):
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        """Train model with validation data"""
        pass

    def predict(self, X):
        """Generate Predictions"""
        return self.model.predict(X)
    
    def save(self, path: Path | str) -> None:
        path = Path(path)
        version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = path / self.config.model_type / version
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.joblib"
        config_path = path / "config.json"
        joblib.dump(self.model, model_path)
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
        

    def log_additional_info(self):
        """Log model-specific information to Neptune"""
        pass

    def log_metrics(self, run, X, y, prefix="val"):
        """Log shared metrics"""
        predicted = self.predict(X)
        run[f"{prefix}/rmse"] = root_mean_squared_error(y, predicted)
        run[f"{prefix}/pearson"] = pearsonr(y, predicted)[0]


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
    def create(cls, config: dict | BaseModelConfig) -> BaseModel:
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
