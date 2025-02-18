import neptune.utils
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from io import StringIO

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GroupKFold

import neptune
from neptune.integrations.xgboost import NeptuneCallback
from neptune.types import File
from neptune.utils import stringify_unsupported

import matplotlib.pyplot as plt

# import custom scripts
from perovskiteml import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, DataSet

###--- Initialization of the Model run ---###
# Initialize the neptune run with the project name and api token
run = neptune.init_run(
    project=NEPTUNE_PROJECT,
    api_token=NEPTUNE_API_TOKEN,
    tags=["xgboost", "shap", "v2.0"],
)
# Create the neptune callback function to log our run
neptune_callback = NeptuneCallback(
    run=run,
    log_importance=False,
    log_tree=None
)

# Set seed to keep results reproducible
seed = 42
run['seed'] = seed # logs the seed in neptune

# Preprocessor and model parameters.
parameters = {
    'dataset': {
        'target': "JV_default_PCE",
        'group_by': "Cell_stack_sequence"
    },
    'preprocessor': {
        'threshold': 0.25,
        'exclude_sections': [
            "Reference information",
            "Cell definition",
            "Outdoor testing",
            "JV data",
        ],
        'exclude_cols': []
    },
    'model': {
        'objective': "reg:squarederror",
        'eval_metric': ["mae", "rmse"],
        'max_depth': 6,
        'eta': 0.3, # Learning rate
    } # Model parameters for XGBoost. More parameters can be added.
    # https://xgboost.readthedocs.io/en/stable/parameter
}
run['parameters'] = stringify_unsupported(parameters) # logs the parameters in neptune
num_round = 300 # Number iterations for the training algorithm.
early_stopping_rounds = None # Number of rounds to check for improvements before stopping
# Value of None disables early stopping

###--- Initialize the Preprocessor ---###

# Load the DataSet
dataset = DataSet(**parameters['dataset'])

# Generate preprocessed data
X, y = dataset.preprocess(**parameters['preprocessor'])

# Define the preprocessor
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value', 
    unknown_value=-1
) # Encodes categorical data using an ordinal encoder.
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder

numerical_selector = make_column_selector(
    dtype_include=np.number
) # Selects the numerical features.
categorical_selector = make_column_selector(
    dtype_include=[bool, object]
) # Selects the categorical features.

preprocessor = ColumnTransformer([
    ('numerical', 'passthrough', numerical_selector), # Allows numerical features to pass.
    ('categorical', encoder, categorical_selector), # Encodes categorical features.
]) # Transforms numerical and categorical data separately.

# The preprocessor creates a transformed version of the data without human readable feature name. Replace these with the original feature names.
all_columns = list(X.select_dtypes(np.number).columns) + list(X.select_dtypes([bool, object]).columns) # Create a list of feature names.

X_transformed = pd.DataFrame(preprocessor.fit_transform(X, y)) # Transform data
X_transformed.columns = all_columns # Replace column names

# Create the training and testing split of data
gkf = GroupKFold(n_splits=2)
train, test = next(gkf.split(X_transformed, y, groups=dataset.groups))
X_train, X_test = X_transformed.loc[train], X_transformed.loc[test]
y_train, y_test = y[train], y[test]

###--- Define the Model ---###
# https://xgboost.readthedocs.io/en/stable/python/python_intro.html
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)
evals = [(dtrain, 'train'), (dval, 'valid')] # evaluations that log to neptune.
model = xgb.train(
    params=parameters['model'],
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    callbacks=[
        neptune_callback,
    ]
) # Trains the model and logs to neptune.

# Generate predictions from the trained model.
y_pred = model.predict(dval)
pred_actual = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred
})
csv_buffer = StringIO()
pred_actual.to_csv(csv_buffer, index=False)
run['training/data/predictions'].upload(File.from_stream(csv_buffer, extension="csv"))

# Generate shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Plotting
fig = plt.figure()
shap.summary_plot(
    shap_values[:1000, :],
    X_transformed.iloc[:1000, :],
    plot_size=[12, 6.75],
    show=False,
)
run['plots/beeswarm'].upload(fig)

fig = plt.figure()
shap.summary_plot(
    shap_values,
    X_transformed,
    plot_type="bar",
    plot_size=[12, 6.75],
    show=False,    
)
run['plots/bar'].upload(fig)

run.stop()