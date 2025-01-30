# --- How to run ---
# cd src
# python -m models.kneighbor-classifier
# ------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate

import scipy.stats as stats

# import custom scripts
from utils import *

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
    #   "Reference information",
      "Cell definition",
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
mask = DATABASE.data[TARGET_COL].notna()
target_data = DATABASE.data[mask][TARGET_COL]

ci_percent = 1/3
params = stats.skewnorm.fit(target_data)
ci = stats.skewnorm.interval(ci_percent, *params)

def quality(data, ci):
  if data < ci[0]:
    return 'poor'
  elif ci[0] <= data < ci[1]:
    return 'moderate'
  else:
    return 'good'

target = target_data.apply(lambda x: quality(x, ci))
target

# Initialize model pipeline
# selector['numerical'].remove(TARGET_COL)

numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='constant', fill_value='none')

categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_pipeline = Pipeline([
    ('imputer', categorical_imputer),
    ('encoder', categorical_encoder)
])


preprocessor = ColumnTransformer([
    ('numerical', numerical_imputer, selector['numerical']),
    ('categorical', categorical_pipeline, selector['categorical']),
    ('patterned', categorical_pipeline, selector['patterned'])
])


scaler = StandardScaler().set_output(transform="pandas")
clf = KNeighborsClassifier(n_neighbors=20)

model = make_pipeline(preprocessor, scaler, clf)

# data = data.drop(columns=[TARGET_COL])[mask]
data = data[mask]
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.30, random_state=42
)
print(X_train.head())
print(y_train.head())

model.fit(X_train, y_train)
x = model.predict(X_test) == y_test
print(
  "Accuracy: "
  f"{x.mean():.3f}"
)


cv_results = cross_validate(model, data, target, cv=10)
scores = cv_results["test_score"]
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)