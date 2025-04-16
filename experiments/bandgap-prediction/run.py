import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from perovskiteml.experiments import ExperimentConfig
from perovskiteml.data import Dataset
from perovskiteml.preprocessing import Preprocessor, PrunerFactory
from perovskiteml.models import ModelFactory
from perovskiteml.validation import Validator
from perovskiteml.logging import neptune_context
from perovskiteml.plotting import plot_actual_vs_predicted, plot_feature_importance
from perovskiteml.utils import load_config, config_to_neptune_format
from sklearn.model_selection import train_test_split, cross_val_score


def run(config_path):
    config_dict = load_config(config_path)
    config = ExperimentConfig.load(config_path)
    time_initialized = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = Path("results") / config.name / time_initialized

    with neptune_context(config.logging) as neptune_run:
        if neptune_run:
            neptune_run["config"] = config_to_neptune_format(config)
            neptune_run["config/file"].upload(config_path)

        dataset = Dataset.load(config.data.path)
        
        if config.pruning:
            pruner = PrunerFactory.create(config.pruning)
            pruner.prune(dataset)
        
        X, y = dataset.split_target(config.data.target_feature)
        
        if config.process:
            preprocessor = Preprocessor(config.process)
            X = preprocessor.preprocess(X, y)

        validator = Validator(config.validation) if config.validation else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.test_size, random_state=config.seed
        )
        
        model = ModelFactory.create(config.model)
        model.train(
            X_train, y_train, X_val, y_val,
            cv=validator,
            run=neptune_run
        )

        y_val_pred = model.predict(X_val)
        y_train_pred = model.predict(X_train)

        residual_plot = plot_actual_vs_predicted(
            config.model.model_type, y_train, y_train_pred, y_val, y_val_pred
        )
        importance_plot = plot_feature_importance(model, X)

        if neptune_run:
            neptune_run["config"] = config_to_neptune_format(config_dict)
            neptune_run["config/file"].upload(config_path)
            neptune_run["plots/residual"].upload(residual_plot)
            neptune_run["plots/importance"].upload(importance_plot)
        
        # Local Saving
        if config.local_save:
            figure_path = results_path / "figures"
            os.makedirs(figure_path, exist_ok=True)
            model.save(results_path)
            residual_plot.savefig(figure_path / "residual.svg")
            importance_plot.savefig(figure_path / "importance.svg")


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
