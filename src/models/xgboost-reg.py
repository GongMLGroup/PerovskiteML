# --- How to run ---
# cd .\src\models\
# python .\xgboost-reg.py
# ------------------
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
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
mask = DATASET.data[TARGET_COL].notna()
X = data[mask]
y = DATASET.data[mask][TARGET_COL]

print("{X.shape[1]} features")
print(X.shape)
print(y.shape)

categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

column_preprocessor = ColumnTransformer([
    ('numerical', 'passthrough', selector['numerical']),
    ('categorical', categorical_encoder, selector['categorical']),
    ('patterned', categorical_encoder, selector['patterned'])
])

scaler = StandardScaler().set_output(transform="pandas")
preprocessor = make_pipeline(column_preprocessor, scaler)

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

X_transformed = model.named_steps['preprocessor'].transform(X)
X_transformed.columns = X.columns

explainer = shap.TreeExplainer(model.named_steps['xgbregressor'])
shap_values = explainer.shap_values(X_transformed)
shap.summary_plot(shap_values[:1000, :], X_transformed.iloc[:1000, :])
shap.summary_plot(shap_values, X_transformed, plot_type="bar")
shap.force_plot(
    explainer.expected_value, shap_values[:1000, :], X.iloc[:1000, :]
)
plt.show()