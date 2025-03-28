import os
import sys
import neptune.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from perovskiteml.data.base import BaseDataset
from perovskiteml.utils.config_parser import load_config, config_to_neptune_format
from perovskiteml.models import ModelFactory
from perovskiteml.models.base import BaseModelHandler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def init_neptune(config: dict) -> Optional[neptune.Run]:
    if not config["experiment"]["neptune"]:
        return None

    api_token = os.path.expandvars(config["logging"]["api_token"])

    run = neptune.init_run(
        project=config["logging"]["project"],
        api_token=api_token,
        tags=config["logging"]["tags"]
    )
    run["sys/group_tags"].add(config["logging"]["group_tags"])
    return run


def log_metrics(
    run: neptune.Run,
    y: pd.Series,
    y_pred: pd.Series,
    id: str = ""
):

    r = np.corrcoef(y.squeeze(), y_pred.squeeze())[0, 1]
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nR2 on {id} Set:", r2)
    print(f"R value on {id} Set:", r)
    print(f"MAE on {id} Set:", mae)
    print(f"MSE on {id} Set:", mse)
    print(f"RMSE on {id} Set:", rmse)

    if run:
        run[f"metrics/{id}/r"] = r
        run[f"metrics/{id}/r2"] = r2
        run[f"metrics/{id}/mae"] = mae
        run[f"metrics/{id}/mse"] = mse
        run[f"metrics/{id}/rmse"] = rmse


def plot_actual_vs_predicted(
    run: neptune.Run,
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

    if config["experiment"]["local_save"]:
        os.makedirs("src/perovskiteml/results/bandgap_prediction/figures/", exist_ok=True)
        plt.savefig('src/perovskiteml/results/bandgap_prediction/figures/residual.png')
    if run:
        run["plots/residual"].upload(fig)
        
    plt.show()


def plot_feature_importance(
    run: neptune.Run,
    config: dict,
    model:BaseModelHandler,
    X: pd.DataFrame
):
    feature_importance = model.model.feature_importances_
    feature_names = list(X.columns)
    importance_dict = {'feature':feature_names, 'feature_importance':feature_importance}
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.sort_values("feature_importance", ascending=False).reset_index()
    
    fig = plt.figure(figsize=(4.5,4))
    sns.barplot(data=importance_df.head(30),x='feature_importance',y='feature')
    plt.title(f'Feature importance Barplot for {model.config.model_type.upper()}')
    
    if config["experiment"]["local_save"]:
        os.makedirs("src/perovskiteml/results/bandgap_prediction/figures/", exist_ok=True)
        plt.savefig('src/perovskiteml/results/bandgap_prediction/figures/importance.png')
    if run:
        run["plots/importance"].upload(fig)
    
    plt.show()
    

def run(config_path):
    config = load_config(config_path)
    neptune_run = init_neptune(config)
    if neptune_run:
        neptune_run["config"] = config_to_neptune_format(config)
        neptune_run["config/file"].upload(config_path)

    dataset = BaseDataset.load(config["data"]["path"])
    model = ModelFactory.create(config["model"])

    print(dataset.data.head())

    X, y = dataset.split_target(config["data"]["target_feature"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=config["experiment"]["seed"]
    )

    model.fit(X_train, y_train, X_val, y_val)

    mse_scores = cross_val_score(
        estimator=model.model,
        X=X_train, y=y_train,
        cv=5, scoring="neg_mean_squared_error"
    )
    rmse_scores = np.sqrt(-mse_scores)
    print("\nRMSE: {:.4f}".format(rmse_scores.mean()))
    print("Standard Deviation: {:.4f}".format(rmse_scores.std()))

    y_val_pred = model.predict(X_val)
    y_val_pred = y_val_pred.reshape(len(y_val_pred), 1)
    log_metrics(neptune_run, y_val, y_val_pred, id="val")

    y_train_pred = model.predict(X_train)
    y_train_pred = y_train_pred.reshape(len(y_train_pred), 1)
    log_metrics(neptune_run, y_train, y_train_pred, id="train")
    
    plot_actual_vs_predicted(neptune_run, config, y_train, y_train_pred, y_val, y_val_pred)
    plot_feature_importance(neptune_run, config, model, X)

    if config["experiment"]["local_save"]:
        model.save("src/perovskiteml/results/bandgap_prediction/model")


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
