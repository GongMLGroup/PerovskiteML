# --- How to run ---
# cd src\models\
# python xgboost-reg.py
# ------------------
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from io import StringIO

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split

import neptune
from neptune.integrations.xgboost import NeptuneCallback
from neptune.types import File

import matplotlib.pyplot as plt

# import custom scripts
from scripts import *

run = neptune.init_run(
    project=NEPTUNE_PROJECT,
    api_token=NEPTUNE_API_TOKEN
)
neptune_callback = NeptuneCallback(
    run=run,
    log_importance=False,
    log_tree=[0,1,2,3]
)

seed = 42
parameters = {
    'preprocessor': {
        'target': "JV_default_PCE",
        'threshold': 0.75,
        'depth': 0.75,
        'exclude_sections': [
            "Reference information",
            "Cell definition",
            "Outdoor testing",
            "JV data"
        ],
        'exclude_cols': [
            "Outdoor_time_start",
            "Outdoor_time_end"
        ]
    },
    'model': {
        'objective': "reg:squarederror",
        'eval_metric': ["mae", "rmse"]
    }
}
run['seed'] = seed
run['parameters'] = parameters

##--- Process target data ---##
X, y = preprocess_data(**parameters['preprocessor'], verbose=False)
print(f"{X.shape[1]} features")

# Define the preprocessor
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
numerical_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=[bool, object])
preprocessor = ColumnTransformer([
    ('numerical', 'passthrough', numerical_selector),
    ('categorical', encoder, categorical_selector),
])

# Transform data to have named Columns
all_columns = list(X.select_dtypes(np.number).columns) + list(X.select_dtypes([bool, object]).columns)
X_transformed = pd.DataFrame(preprocessor.fit_transform(X, y))
X_transformed.columns = all_columns
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.30, random_state=seed
)

##--- Define the Model ---##
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)
evals = [(dtrain, 'train'), (dval, "valid")]
num_round = 300
model = xgb.train(
    params=parameters['model'],
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evals,
    callbacks=[
        neptune_callback,
        xgb.callback.EarlyStopping(rounds=30)
    ]
)

# generate predictions
y_pred = model.predict(dval)
pred_actual = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred
})
csv_buffer = StringIO()
pred_actual.to_csv(csv_buffer, index=False)
run['training/data/predictions'].upload(File.from_stream(csv_buffer, extension="csv"))

# generate shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# plotting
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