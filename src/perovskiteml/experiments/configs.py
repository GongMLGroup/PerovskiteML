from pathlib import Path
from pydantic import BaseModel, Field
from ..data import DatasetConfig
from ..models import ModelFactory, ModelConfig
from ..preprocessing import PreprocessorConfig, PrunerConfig, PrunerFactory
from ..logging import NeptuneConfig
from ..utils import OptunaSweepConfig, load_config

class ExperimentConfig(BaseModel):
    name: str = "experiment"
    seed: int = 42
    local_save: bool = False
    verbose: int = 0
    data: DatasetConfig
    model: ModelConfig
    pruning: PrunerConfig | None = None
    process: PreprocessorConfig | None = None
    hyperparameters: OptunaSweepConfig | None = None
    logging: NeptuneConfig = Field(default_factory=NeptuneConfig)
    
    @classmethod
    def load(cls, config_path: str | Path):
        config_dict = load_config(config_path)
        return cls.create_from_dict(config_dict)
        
    @classmethod
    def create_from_dict(cls, config_dict: dict):
        experiment_config = config_dict["experiment"]
        data_config = config_dict["data"]
        model_config = ModelFactory._validate_dict_config(config_dict["model"])
        
        pruning_dict = config_dict.get("pruning")
        if pruning_dict:
            pruning_config = PrunerFactory._validate_dict_config(pruning_dict)
        else:
            pruning_config = None
        
        process_config = config_dict.get("process")
        hyperparameter_config = config_dict.get("hyperparameters")
        logging_config = config_dict.get("logging", {})
        
        return cls(
            **experiment_config,
            data = data_config,
            model = model_config,
            pruning = pruning_config,
            process = process_config,
            hyperparameters = hyperparameter_config,
            logging = logging_config
        )
        
        
        
        
        