import os
import sys
import neptune.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from perovskiteml.data.base import BaseDataset
from perovskiteml.utils.config_parser import load_config, config_to_neptune_format
from perovskiteml.logging import neptune_context
from perovskiteml.models import ModelFactory
from perovskiteml.preprocessing.preprocessor import Preprocessor
from perovskiteml.models.base import BaseModelHandler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def log_artifacts(config, config_path, run):
    if run:
        run["config"] = config_to_neptune_format(config)
        run["config/file"].upload(config_path)



def run(config_path):
    config = load_config(config_path)
    time_initialized = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = Path("results") / \
        config["experiment"]["name"] / time_initialized

    with neptune_context(config["logging"]) as neptune_run:
        if neptune_run:
            neptune_run["config"] = config_to_neptune_format(config)
            neptune_run["config/file"].upload(config_path)

        dataset = BaseDataset.load(config["data"]["path"])
        preprocessor = Preprocessor(config["process"])
        model = ModelFactory.create(config["model"])

        X, y = dataset.split_target(config["data"]["target_feature"])
        X_transformed = preprocessor.preprocess(X, y)

        X_train, X_val, y_train, y_val = train_test_split(
            X_transformed, y, test_size=0.30, random_state=config["experiment"]["seed"]
        )

        model.train(X_train, y_train, X_val, y_val, neptune_run)

        if not neptune_run:
            mse_scores = cross_val_score(
                estimator=model.model,
                X=X_train, y=y_train,
                cv=5, scoring="neg_mean_squared_error"
            )
            rmse_scores = np.sqrt(-mse_scores)
            print("\nRMSE: {:.4f}".format(rmse_scores.mean()))
            print("Standard Deviation: {:.4f}".format(rmse_scores.std()))

        y_val_pred = model.predict(X_val)
        y_train_pred = model.predict(X_train)

        residual_plot = plot_actual_vs_predicted(
            config, y_train, y_train_pred, y_val, y_val_pred
        )
        importance_plot = plot_feature_importance(model, X)

        log_artifacts(config, config_path, neptune_run)
        if neptune_run:
            neptune_run["plots/residual"].upload(residual_plot)
            neptune_run["plots/importance"].upload(importance_plot)
        
        # Local Saving
        if config["experiment"]["local_save"]:
            figure_path = results_path / "figures"
            os.makedirs(figure_path, exist_ok=True)
            model.save(results_path)
            residual_plot.savefig(figure_path / "residual.svg")
            importance_plot.savefig(figure_path / "importance.svg")


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
