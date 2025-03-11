import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from perovskiteml.data import ExpandedDataset
from perovskiteml.preprocessing import PrunerFactory
from perovskiteml.utils.config_parser import load_config
from perovskiteml.models import ModelFactory
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector


# This will be replaced with a similar workflow to the pruners and models
def preprocess_data(X, y, config):
    # Define the preprocessor
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )  # Encodes categorical data using an ordinal encoder.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder

    numerical_selector = make_column_selector(
        dtype_include=np.number
    )  # Selects the numerical features.
    categorical_selector = make_column_selector(
        dtype_include=[bool, object]
    )  # Selects the categorical features.

    preprocessor = ColumnTransformer([
        # Allows numerical features to pass.
        ('numerical', 'passthrough', numerical_selector),
        # Encodes categorical features.
        ('categorical', encoder, categorical_selector),
    ])  # Transforms numerical and categorical data separately.

    # The preprocessor creates a transformed version of the data without human readable feature name. Replace these with the original feature names.
    # Create a list of feature names.
    all_columns = list(X.select_dtypes(np.number).columns) + list(X.select_dtypes([bool, object]).columns)

    X_transformed = pd.DataFrame(
        preprocessor.fit_transform(X, y))  # Transform data
    X_transformed.columns = all_columns  # Replace column names

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=0.30, random_state=config["experiment"]["seed"]
    )
    return X_train, X_val, y_train, y_val


def run(config_path):
    config = load_config(config_path)
    dataset = ExpandedDataset.cache_or_compute(config["data"])
    pruner = PrunerFactory.create(config["pruning"])
    model = ModelFactory.create(config["model"])
    
    pruner.prune(dataset)
    X, y = dataset.split_target()
    X_train, X_val, y_train, y_val = preprocess_data(X, y, config)
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Analysis
    plt.axline((0, 0), slope=1, color='k', linestyle='--')
    plt.scatter(y_train, model.predict(X_train))
    plt.scatter(y_val, model.predict(X_val))
    plt.show()


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
