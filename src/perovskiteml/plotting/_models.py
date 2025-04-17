import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.base import BaseModelHandler

def plot_actual_vs_predicted(
    model_type: str,
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
    plt.title(
        f'{model_type.upper()} Model: Actual vs Predicted Bandgap'
    )
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    model: BaseModelHandler,
    X: pd.DataFrame
):
    feature_importance = model.model.feature_importances_
    feature_names = list(X.columns)
    importance_dict = {'feature': feature_names,
                       'feature_importance': feature_importance}
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.sort_values(
        "feature_importance", ascending=False).reset_index()

    fig = plt.figure(figsize=(4.5, 4))
    sns.barplot(data=importance_df.head(20),
                x='feature_importance', y='feature')
    plt.title(
        f'Feature importance Barplot for {model.config.model_type.upper()}')

    return fig