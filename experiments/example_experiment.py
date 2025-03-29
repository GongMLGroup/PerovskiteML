import os
import sys
import shap
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from perovskiteml.data import ExpandedDataset
from perovskiteml.preprocessing.preprocessor import Preprocessor
from perovskiteml.preprocessing import PrunerFactory
from perovskiteml.utils.config_parser import load_config, config_to_neptune_format
from perovskiteml.logging import neptune_context
from perovskiteml.models import ModelFactory
from sklearn.model_selection import train_test_split


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

        dataset = ExpandedDataset.cache_or_compute(config["data"])
        pruner = PrunerFactory.create(config["pruning"])
        preprocessor = Preprocessor(config["process"])
        model = ModelFactory.create(config["model"])

        pruner.prune(dataset)
        X, y = dataset.split_target()
        X_transformed = preprocessor.preprocess(X, y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_transformed, y, test_size=0.30, random_state=config["experiment"]["seed"]
        )
        model.train(X_train, y_train, X_val, y_val, neptune_run)

        # Analysis
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_transformed)

        # Beeswarm Plot
        fig0 = plt.figure()
        shap.summary_plot(
            shap_values[:1000, :],
            X_transformed.iloc[:1000, :],
            plot_size=[12, 6.75],
            show=False,
        )

        # Importance Plot
        fig1 = plt.figure()
        shap.summary_plot(
            shap_values,
            X_transformed,
            plot_type="bar",
            plot_size=[12, 6.75],
            show=False,
        )

        log_artifacts(config, config_path, neptune_run)
        if neptune_run:
            neptune_run['plots/beeswarm'].upload(fig0)
            neptune_run['plots/bar'].upload(fig1)

        # Local Saving
        if config["experiment"]["local_save"]:
            figure_path = results_path / "figures"
            shap_path = results_path / "shap"
            os.makedirs(figure_path, exist_ok=True)
            os.makedirs(shap_path, exist_ok=True)
            model.save(results_path)
            fig0.savefig(figure_path / "beeswarm.svg")
            fig1.savefig(figure_path / "importance.svg")
            joblib.dump(explainer, shap_path / "explainer.joblib", compress=3)
            joblib.dump(shap_values, shap_path / "shap_values.joblib", compress=3)


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
