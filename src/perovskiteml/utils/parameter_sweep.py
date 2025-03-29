import optuna
from pydantic import BaseModel, ConfigDict
from typing import Literal, Any

def _is_list_of_dicts(value:list) -> bool:
    return all(isinstance(item, dict) for item in value)

class ParameterSweepConfig(BaseModel):
    type: Literal["float", "int"]
    model_config = ConfigDict(extra="allow")


class ParameterSweepHandler:
    def __init__(self, config: dict):
        self.parameters = self._generate_config(config)

    @classmethod
    def _generate_config(cls, config: dict | list | Any):
        """Uses the config structure to recusively create Sweeps for parameters"""
        if isinstance(config, dict):
            return {section_name: {
                name: cls._generate_params(param)
                for name, param in section.items()
            } for section_name, section in config.items()}
        elif isinstance(config, list) and _is_list_of_dicts(config):
            return [{
                name: cls._generate_params(param)
                for name, param in step.items()
            } for step in config]
            
        return config

    @classmethod
    def _generate_params(cls, param: dict | list | Any):
        """
        Generates the ParameterSweepConfig
        recursively calls _generate_config for lists of parameters
        """
        if isinstance(param, dict):
            return ParameterSweepConfig(**param)
        elif isinstance(param, list):
            return cls._generate_config(param)
        
        return param

    @staticmethod
    def _suggest_param(trial: optuna.Trial, name: str, config: ParameterSweepConfig):
        """Selects the appropriate optuna parameter suggestion method"""
        match config.type:
            case "float":
                return trial.suggest_float(
                    name, **config.model_dump(exclude="type")
                )
            case "int":
                return trial.suggest_int(
                    name, **config.model_dump(exclude="type")
                )
            case "categorical":
                return trial.suggest_categorical(
                    name, **config.model_dump(exclude="type")
                )

    @classmethod
    def _suggested_config(cls, trial: optuna.Trial, config, name=""):
        """Recursively copy the config with values suggested from optuna"""
        if isinstance(config, dict):
            return {
                key: cls._suggested_config(trial, value, key)
                for key, value in config.items()
            }

        elif isinstance(config, list):
            return [
                cls._suggested_config(trial, value)
                for value in config
            ]

        elif isinstance(config, ParameterSweepConfig):
            return cls._suggest_param(trial, name, config)
        
        return config

    def suggested_config(self, trial: optuna.Trial) -> dict:
        """Returns a config with suggested parameters from the given optuna trial"""
        return self._suggested_config(trial, self.parameters)