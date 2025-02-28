from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Annotated
from .reduction import prune_and_combine
# from ..data import DataSet


class BasePrunerConfig(BaseModel):
    method: Literal["breadth_pruner", "depth_pruner"]
    sparsity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    remove: dict[str, list] = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)


class DepthParams(BaseModel):
    device_layer_coverage: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75


class PrunerConfig(BasePrunerConfig):
    @model_validator(mode="after")
    def validate_method_params(self):
        method = self.method
        parameters = self.params.get(method, {})

        match method:
            case "breadth_pruner":
                self.params = parameters
            case "depth_pruner":
                self.params = DepthParams(**parameters)
            case _:
                raise ValueError(f"Invalid method: {method}")

        return self


class PrunerFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(pruner_class):
            cls._registry[name] = pruner_class
            return pruner_class
        return decorator

    @classmethod
    def create(cls, config: BasePrunerConfig):
        return cls._registry[config.method](config)


class BasePruner(ABC):
    def __init__(self, config: PrunerConfig):
        self.config = config

    @abstractmethod
    def prune(self, dataset):
        pass
    

@PrunerFactory.register("breadth_pruner")
class BreadthPruner(BasePruner):
    # TODO: add functionality
    def prune(self, dataset):
        if not dataset.features:
            dataset.get_dataset()
        if self.config.remove:
            dataset.remove(
                sections=self.config.remove.get("sections", []),
                features=self.config.remove.get("features", [])
            )
        if self.config.sparsity_threshold > 0.0:
            dataset.prune_by_sparsity(self.config.sparsity_threshold)
        return dataset


@PrunerFactory.register("depth_pruner")
class DepthPruner(BasePruner):
    # TODO: add functionality
    def prune(self, dataset):
        if self.config.params.device_layer_coverage < 1.0:
            if not dataset.features:
                dataset.get_dataset()
            dataset.features = prune_and_combine(
                dataset.features,
                self.config.params.device_layer_coverage
            )
            
        return dataset
