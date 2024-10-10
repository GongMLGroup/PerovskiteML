# --- How to run ---
# cd src\models\
# python xgboost-reg.py
# ------------------
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

# import custom scripts
from scripts import *

# Threshold (%) of data for a feature to be included
THRESHOLD = 0.75
# Depth Threshold (%) of data included from layer data.
# --- picks the maximum depth that includes the threshold % of devices
DEPTH_THRESHOLD = 0.75

TARGET_COL = "JV_default_PCE"

data, selector = preprocess_data(
    THRESHOLD,
    DEPTH_THRESHOLD,
    exclude_sections=[
      "Reference information",
      "Cell definition",
      "Outdoor testing",
    #   "Additional layers",
      "JV data"
      ],
    exclude_cols=[
      "Outdoor_time_start",
      "Outdoor_time_end"
      ],
    verbose=False
)

# Process target data
DATASET.load_data()
mask = DATASET.data[TARGET_COL].notna()
X = data[mask]
y = DATASET.data[mask][TARGET_COL]

print("{X.shape[1]} features")
print(X.shape)
print(y.shape)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

numerical_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=[bool, object])

preprocessor = ColumnTransformer([
    ('numerical', 'passthrough', numerical_selector),
    ('categorical', encoder, categorical_selector),
])

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('xgbregressor', xgb_model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}, RMSE: {mse**0.5:.2f}")

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
display_scores(-scores)

X_transformed = pd.DataFrame(model.named_steps['preprocessor'].transform(X))
_cols_transformed = model.named_steps['preprocessor'].get_feature_names_out(X.columns)
cols_transformed = [string.split('__')[1] for string in _cols_transformed]
X_transformed.columns = cols_transformed

explainer = shap.TreeExplainer(model.named_steps['xgbregressor'])
shap_values = explainer.shap_values(X_transformed)
shap.summary_plot(shap_values[:1000, :], X_transformed.iloc[:1000, :])
shap.summary_plot(shap_values, X_transformed, plot_type="bar")
shap.force_plot(
    explainer.expected_value, shap_values[:1000, :], X.iloc[:1000, :]
)
plt.show()