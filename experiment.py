import os
import sys
import neptune.utils
import shap
import pandas as pd
import numpy as np
import neptune
import matplotlib.pyplot as plt
from typing import Optional
from perovskiteml.data import ExpandedDataset
from perovskiteml.data.base import BaseDataset
from perovskiteml.preprocessing.preprocessor import Preprocessor
from perovskiteml.preprocessing import PrunerFactory
from perovskiteml.utils.config_parser import load_config, config_to_neptune_format
from perovskiteml.models import ModelFactory
from sklearn.model_selection import train_test_split

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

def run(config_path):
    config = load_config(config_path)
    neptune_run = init_neptune(config)
    if neptune_run:
        neptune_run["config"] = config_to_neptune_format(config)
        neptune_run["config/file"].upload(config_path)
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

    model.fit(X_train, y_train, X_val, y_val)
    model.save("src/perovskiteml/results/models")

    # Analysis
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_transformed)
    fig = plt.figure()
    shap.summary_plot(
        shap_values[:1000, :],
        X_transformed.iloc[:1000, :],
        plot_size=[12, 6.75],
        show=False,
    )
    if neptune_run:
        neptune_run['plots/beeswarm'].upload(fig)
    
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        plot_size=[12, 6.75],
        show=False,    
    )
    if neptune_run:
        neptune_run['plots/bar'].upload(fig)
        neptune_run.stop()


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
