import sys
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perovskiteml.data import ExpandedDataset
from perovskiteml.preprocessing.preprocessor import Preprocessor
from perovskiteml.preprocessing import PrunerFactory
from perovskiteml.utils.config_parser import load_config
from perovskiteml.models import ModelFactory
from sklearn.model_selection import train_test_split

def run(config_path):
    config = load_config(config_path)
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
        show=True,
    )
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        plot_size=[12, 6.75],
        show=True,    
    )


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
