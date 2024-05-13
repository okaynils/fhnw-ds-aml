import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve)

def add_prefix_except_id(df, prefix, id_exceptions=[]):
    """
    Adds a prefix to all column names in the DataFrame except those containing '_id' or 'id_', 
    with an exception for any column names explicitly listed in id_exceptions.

    Parameters:
    - df: pandas.DataFrame to be modified.
    - prefix: String prefix to be added.
    - id_exceptions: List of column names that contain '_id' or 'id_' but should still have the prefix added.

    Returns:
    - A new DataFrame with updated column names.
    """
    # Define a new DataFrame to avoid modifying the original one
    new_df = df.copy()
    
    # Rename columns, adding prefix based on the conditions
    new_df.columns = [
        f'{prefix}{col}' if (('_id' not in col and 'id_' not in col) or col in id_exceptions) else col 
        for col in df.columns
    ]
    
    return new_df

from sklearn.base import BaseEstimator, TransformerMixin

class DateToUnixTimestampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X, y=None):
        # Assuming 'Date' is the column to be transformed
        X_transformed = X.copy()
        X_transformed['Date'] = X_transformed['Date'].astype('int64') // 10**9
        return X_transformed

def create_pipeline(categorical_features, numerical_features, estimator: BaseEstimator):
    """
    Creates a pipeline with specified categorical and numerical features.
    
    Parameters:
    - categorical_features: list of strings, names of the categorical columns
    - numerical_features: list of strings, names of the numerical columns
    
    Returns:
    - A Scikit-Learn Pipeline object configured for the specified features.
    """
    categorical_preprocessor = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])
    
    numerical_preprocessor = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('cats', categorical_preprocessor, categorical_features),
        ('nums', numerical_preprocessor, numerical_features)
    ], remainder='passthrough')
    
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', estimator)
    ])
    
    return pipeline


def evaluate_model(pipeline, X, y, model_name=None):
    # Configure Seaborn aesthetics
    sns.set(style="whitegrid")

    # Determine model name
    if not model_name:
        # Attempt to get the class name of the last step in the pipeline (typically the estimator)
        model_name = pipeline.steps[-1][1].__class__.__name__ if pipeline.steps else "Estimator"

    # Setup Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validated predictions for metrics and probabilities
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    y_prob = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted'),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1 Score': f1_score(y, y_pred, average='weighted'),
        'ROC AUC': roc_auc_score(y, y_prob[:, 1])
    }

    # Create ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(x=fpr, y=tpr, label=f'{model_name} ROC (area = {metrics["ROC AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')

    # Create Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_prob[:, 1])
    plt.subplot(1, 2, 2)
    sns.lineplot(x=recall, y=precision, label=f'{model_name} Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.show()

    # Create a lift curve
    skplt.metrics.plot_lift_curve(y, y_prob)
    plt.title(f'{model_name} Lift Curve')
    plt.show()

    # Return metrics as a DataFrame for easy viewing
    return pd.DataFrame(metrics, index=[0])