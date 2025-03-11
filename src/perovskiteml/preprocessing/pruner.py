from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union
from .reduction import prune_and_combine, prune_by_sparsity
from ..data.base import BaseDataset


# --------------------------
# Base Model Definition
# --------------------------


class BasePrunerConfig(BaseModel, ABC):
    """Base configuration for all pruners"""
    method: str


class BasePruner(ABC):
    def __init__(self, config: BasePrunerConfig):
        self.config = config

    @abstractmethod
    def prune(self, dataset: BaseDataset) -> BaseDataset:
        """Prune input data according to config"""
        dataset.metadata.processing_history.append(self.config)
        return dataset


class PrunerFactory:
    _pruners = {}
    _configs = {}

    @classmethod
    def register_pruner(cls, name: str):
        def decorator(pruner_cls):
            cls._pruners[name] = pruner_cls
            return pruner_cls
        return decorator

    @classmethod
    def register_config(cls, name: str):
        def decorator(config_cls):
            cls._configs[name] = config_cls
            return config_cls
        return decorator

    @classmethod
    def create(cls, config: Union[dict, BasePrunerConfig]) -> BasePruner:
        """Create pruner from validated configuration"""
        if isinstance(config, dict):
            config = cls._validate_dict_config(config)

        if config.method not in cls._pruners:
            raise ValueError(
                f"Unregistered pruner type: {config.method}. "
                f"Registered pruners include: {list(cls._pruners.keys())}"   
            )

        return cls._pruners[config.method](config)

    @classmethod
    def _validate_dict_config(cls, config: dict) -> BasePrunerConfig:
        """Convert and validate dict config to appropriate model"""
        method = config.get("method")

        if method not in cls._configs:
            raise ValueError(
                f"Unregistered pruner type: {method}. "
                f"Registered pruners include: {list(cls._pruners.keys())}"   
            )

        return cls._configs[method](**config)


# --------------------------
# Configuration Models
# --------------------------


@PrunerFactory.register_config("feature_pruner")
class FeaturePrunerConfig(BasePrunerConfig):
    method: Literal["feature_pruner"] = "feature_pruner"
    sections: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)


@PrunerFactory.register_config("breadth_pruner")
class BreadthPrunerConfig(BasePrunerConfig):
    method: Literal["breadth_pruner"] = "breadth_pruner"
    sparsity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25


@PrunerFactory.register_config("depth_pruner")
class DepthPrunerConfig(BasePrunerConfig):
    method: Literal["depth_pruner"] = "depth_pruner"
    layer_coverage: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75


@PrunerFactory.register_config("chain_pruner")
class ChainPrunerConfig(BasePrunerConfig):
    method: Literal["chain_pruner"] = "chain_pruner"
    steps: list[Union[
        FeaturePrunerConfig,
        BreadthPrunerConfig,
        DepthPrunerConfig
    ]] = Field(
        ...,
        min_items=1,
        description="Ordered list of pruning steps"
    )

# --------------------------
# Pruner Implementation
# --------------------------


@PrunerFactory.register_pruner("feature_pruner")
class FeaturePruner(BasePruner):
    def __init__(self, config: FeaturePrunerConfig):
        super().__init__(config)

    def prune(self, dataset: BaseDataset):
        print(f"Pruning sections: {self.config.sections}")
        print(f"Pruning features: {self.config.features}")
        dataset.remove(
            remove_sections=self.config.sections,
            remove_features=self.config.features
        )
        
        super().prune(dataset)
        return dataset


@PrunerFactory.register_pruner("breadth_pruner")
class BreadthPruner(BasePruner):
    def __init__(self, config: BreadthPrunerConfig):
        super().__init__(config)

    def prune(self, dataset: BaseDataset):
        if self.config.sparsity_threshold > 0.0:
            dataset._features = prune_by_sparsity(
                dataset._features,
                self.config.sparsity_threshold
            )
            
        super().prune(dataset)
        return dataset


@PrunerFactory.register_pruner("depth_pruner")
class DepthPruner(BasePruner):
    def __init__(self, config: DepthPrunerConfig):
        super().__init__(config)

    def prune(self, dataset: BaseDataset):
        if self.config.layer_coverage < 1.0:
            dataset._features = prune_and_combine(
                dataset._features,
                self.config.layer_coverage
            )
            
        super().prune(dataset)
        return dataset


@PrunerFactory.register_pruner("chain_pruner")
class ChainPruner(BasePruner):
    def __init__(self, config: ChainPrunerConfig):
        super().__init__(config)
        self.pruners = [PrunerFactory.create(
            step) for step in self.config.steps]

    def prune(self, dataset: BaseDataset):
        for pruner in self.pruners:
            dataset = pruner.prune(dataset)
        return dataset
