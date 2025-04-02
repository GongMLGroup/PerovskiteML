import os
import sys
import optuna
import neptune.integrations.optuna as optuna_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from pathlib import Path
from perovskiteml.data.base import BaseDataset
from perovskiteml.preprocessing.preprocessor import Preprocessor
from perovskiteml.models.base import BaseModelHandler
from perovskiteml.utils.config_parser import load_config, config_to_neptune_format
from perovskiteml.utils.parameter_sweep import ParameterSweepHandler
from perovskiteml.logging import neptune_context
from perovskiteml.models import ModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def log_artifacts(config, config_path, run):
    if run:
        run["config"] = config_to_neptune_format(config)
        run["config/file"].upload(config_path)

def plot_actual_vs_predicted(
    config: dict,
    y_train: pd.Series,
    y_train_pred: pd.Series,
    y_val: pd.Series,
    y_val_pred: pd.Series
):
    # Plotting the results for the training set
    fig = plt.figure(figsize=(4.5, 4))
    plt.plot(y_train, y_train, color='black', label='Actual Values')
    plt.scatter(y_train, y_train_pred, edgecolors='royalblue',
                linewidth=1.2, color='royalblue', label='Train set')
    plt.scatter(y_val, y_val_pred, edgecolors='deeppink',
                linewidth=1.2, color='deeppink', label='Test set')
    plt.title('Training Set: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    # plt.grid(True)
    plt.title(
        f'{config["model"]["model_type"].upper()} Model: Actual vs Predicted Bandgap'
    )
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    model: BaseModelHandler,
    X: pd.DataFrame
):
    feature_importance = model.model.feature_importances_
    feature_names = list(X.columns)
    importance_dict = {'feature': feature_names,
                       'feature_importance': feature_importance}
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.sort_values(
        "feature_importance", ascending=False).reset_index()

    fig = plt.figure(figsize=(4.5, 4))
    sns.barplot(data=importance_df.head(30),
                x='feature_importance', y='feature')
    plt.title(
        f'Feature importance Barplot for {model.config.model_type.upper()}')

    return fig


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
    

def run(config_path):
    config_dict = load_config(config_path)
    time_initialized = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = Path("results") / \
        config_dict["experiment"]["name"] / time_initialized
    study = optuna.create_study(
        direction=config_dict["hyperparameters"]["direction"],
        study_name=config_dict["hyperparameters"]["study_name"]
    )
    config_sweep = ParameterSweepHandler(config_dict)
    dataset = BaseDataset.load(config_dict["data"]["path"])

    def objective(trial, config_sweep=config_sweep, dataset=dataset):
        model, _, split = create_run(trial, config_sweep, dataset)
        model.fit(*split)
        X_train, y_train, _, _ = split
        y_pred = model.predict(X_train)
        return root_mean_squared_error(y_pred, y_train)

    with neptune_context(config_dict["logging"]) as neptune_run:
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
            n_trials=config_dict["hyperparameters"]["n_trials"]
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
            config_dict, y_train, y_train_pred, y_val, y_val_pred
        )
        
        log_artifacts(config_dict, config_path, neptune_run)
        if neptune_run:
            neptune_run["plots/residual"].upload(residual_plot)
            neptune_run["plots/importance"].upload(importance_plot)
        
        # Local Saving
        if config_dict["experiment"]["local_save"]:
            figure_path = results_path / "figures"
            os.makedirs(figure_path, exist_ok=True)
            model.save(results_path)
            residual_plot.savefig(figure_path / "residual.svg")
            importance_plot.savefig(figure_path / "importance.svg")
        
        


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
