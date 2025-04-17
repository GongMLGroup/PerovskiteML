import os
import sys
import optuna
import neptune.integrations.optuna as optuna_utils
import warnings
from datetime import datetime
from pathlib import Path
from perovskiteml.experiments import ExperimentConfig
from perovskiteml.models.base import BaseModelHandler
from perovskiteml.data import Dataset
from perovskiteml.preprocessing import PrunerFactory, Preprocessor
from perovskiteml.validation import Validator
from perovskiteml.models import ModelFactory
from perovskiteml.logging import NeptuneConfig, neptune_context
from perovskiteml.utils import (
    OptunaSweepConfig,
    ParameterSweepHandler,
    load_config,
    config_to_neptune_format
)
from perovskiteml.plotting import plot_actual_vs_predicted, plot_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def create_run(trial, config_sweep, dataset) -> tuple[BaseModelHandler, tuple, tuple]:
    config = config_sweep.suggested_config(trial)
    preprocessor = Preprocessor(config["process"])
    model = ModelFactory.create(config["model"])
    if config["hyperparameters"]["pruning"]:
        model.init_callbacks(trial=trial)

    X, y = dataset.split_target(config["data"]["target_feature"])
    X_transformed = preprocessor.preprocess(X, y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=0.30, random_state=config["experiment"]["seed"]
    )
    
    return model, (X_transformed, y), (X_train, y_train, X_val, y_val)

def create_run(
    trial: optuna.Trial,
    config_sweep: ParameterSweepHandler,
    dataset: Dataset
) -> tuple[BaseModelHandler, Validator | None, tuple, tuple]:
    
    config = config_sweep.suggested_config(trial)
    experiment = ExperimentConfig.create_from_dict(config)
    
    if experiment.pruning:
        pruner = PrunerFactory.create(experiment.pruning)
        pruner.prune(dataset)
        
    X, y = dataset.split_target(experiment.data.target_feature)
    if experiment.process:
        preprocessor = Preprocessor(experiment.process)
        X = preprocessor.preprocess(X, y)
    
    model = ModelFactory.create(experiment.model)
    
    validator = None
    if experiment.validation:
        if experiment.hyperparameters.pruning:
            warnings.warn("Cross Validation does not Support Optuna Pruning: Cross Validation takes priority. Optuna Pruning ignored.")
        validator = Validator(experiment.validation)
    elif experiment.hyperparameters.pruning:
        model.init_callbacks(trial=trial)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=experiment.test_size, random_state=experiment.seed
    )
    
    return model, validator, (X, y), (X_train, y_train, X_val, y_val)
    

def run(config_path):
    config_dict = load_config(config_path)
    experiment_config = config_dict["experiment"]
    logging_config = NeptuneConfig(**config_dict.get("logging", {}))
    optuna_config = OptunaSweepConfig(**config_dict.get("hyperparameters", {}))
    
    time_initialized = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = Path("results") / \
        config_dict["experiment"]["name"] / time_initialized
    study = optuna.create_study(
        direction=config_dict["hyperparameters"]["direction"],
        study_name=config_dict["hyperparameters"]["study_name"]
    )
    config_sweep = ParameterSweepHandler(config_dict)
    dataset = Dataset.load(config_dict["data"]["path"])

    def objective(trial, config_sweep=config_sweep, dataset=dataset):
        model, cv, _, split = create_run(trial, config_sweep, dataset)
        X_train, y_train, _, _ = split
        
        if cv:
            cv.cross_validate(model, X_train, y_train)
            return cv._results["test_rmse"].mean()
        else:
            model.fit(*split)
            y_pred = model.predict(X_train)
            return root_mean_squared_error(y_pred, y_train)

    with neptune_context(logging_config) as neptune_run:
        if neptune_run:
            neptune_callback = optuna_utils.NeptuneCallback(
                neptune_run,
                plots_update_freq=10,
                study_update_freq=10,
                log_all_trials=False
            )
        else:
            neptune_callback = None
        study.optimize(
            objective,
            callbacks=[neptune_callback] if neptune_callback else None,
            n_trials=optuna_config.n_trials
        )
        
        print(f"Rerun Best Model: {study.best_params}")
        model, cv, data, split = create_run(
            study.best_trial, config_sweep, dataset
        )
        model.train(*split, cv=cv, run=neptune_run)
        
        X_train, y_train, X_val, y_val = split
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        importance_plot = plot_feature_importance(model, data[0])
        residual_plot = plot_actual_vs_predicted(
            model.config.model_type, y_train, y_train_pred, y_val, y_val_pred
        )
        
        if neptune_run:
            neptune_run["config"] = config_to_neptune_format(config_dict)
            neptune_run["config/file"].upload(config_path)
            neptune_run["plots/residual"].upload(residual_plot)
            neptune_run["plots/importance"].upload(importance_plot)
        
        # Local Saving
        if experiment_config.get("local_save", False):
            figure_path = results_path / "figures"
            os.makedirs(figure_path, exist_ok=True)
            model.save(results_path)
            residual_plot.savefig(figure_path / "residual.svg")
            importance_plot.savefig(figure_path / "importance.svg")
        
        


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
