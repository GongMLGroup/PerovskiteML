import os
import sys
import optuna
import neptune.integrations.optuna as optuna_utils
from datetime import datetime
from pathlib import Path
from perovskiteml.experiments import ExperimentConfig
from perovskiteml.data import ExpandedDataset
from perovskiteml.models import ModelFactory
from perovskiteml.models.base import BaseModelHandler
from perovskiteml.preprocessing import PrunerFactory, Preprocessor
from perovskiteml.logging import NeptuneConfig
from perovskiteml.plotting import plot_feature_importance, plot_actual_vs_predicted
from perovskiteml.utils import (
    ParameterSweepHandler, OptunaSweepConfig, load_config, config_to_neptune_format
)
from perovskiteml.logging import neptune_context
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def create_run(
    trial: optuna.Trial,
    config_sweep: ParameterSweepHandler,
    dataset: ExpandedDataset
) -> tuple[BaseModelHandler, tuple, tuple]:
    
    config = config_sweep.suggested_config(trial)
    experiment = ExperimentConfig.create_from_dict(config)
    dataset.reset_features()
    
    if experiment.pruning:
        pruner = PrunerFactory.create(experiment.pruning)
        pruner.prune(dataset)
        
    X, y = dataset.split_target()
    if experiment.process:
        preprocessor = Preprocessor(experiment.process)
        X_transformed = preprocessor.preprocess(X, y)
    else:
        X_transformed = X
    
    model = ModelFactory.create(experiment.model)
    if experiment.hyperparameters.pruning:
        model.init_callbacks(trial=trial)

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=0.30, random_state=experiment.seed
    )
    
    return model, (X_transformed, y), (X_train, y_train, X_val, y_val)

def run(config_path):
    config_dict = load_config(config_path)
    experiment_config = config_dict["experiment"]
    logging_config = NeptuneConfig(**config_dict.get("logging", {}))
    sweep_config = OptunaSweepConfig(**config_dict.get("hyperparameters", {}))
    
    time_initialized = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = Path("results") / \
        experiment_config.get("name", "") / time_initialized
    study = optuna.create_study(
        direction=sweep_config.direction,
        study_name=sweep_config.study_name
    )
    config_sweep = ParameterSweepHandler(config_dict)
    dataset = ExpandedDataset.cache_or_compute(**config_dict.get("data"))

    def objective(trial, config_sweep=config_sweep, dataset=dataset):
        model, _, split = create_run(trial, config_sweep, dataset)
        model.fit(*split)
        X_train, y_train, _, _ = split
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
            n_trials=sweep_config.n_trials
        )
        
        print("Rerun Best Model")
        model, data, split = create_run(
            study.best_trial, config_sweep, dataset
        )
        model.train(*split, run=neptune_run)
        
        X_train, y_train, X_val, y_val = split
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        importance_plot = plot_feature_importance(model, data[0])
        residual_plot = plot_actual_vs_predicted(
            config_dict["model"]["model_type"], y_train, y_train_pred, y_val, y_val_pred
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
