import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

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
    
    # Store metrics for each fold
    metrics_list = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics_list.append({
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
            'ROC AUC': roc_auc_score(y_test, y_prob)
        })
    
    # Calculate mean and standard deviation of metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_mean = metrics_df.mean()
    metrics_std = metrics_df.std()

    # Create a DataFrame with mean and std of metrics
    metrics_summary_df = pd.DataFrame({
        'Metric': metrics_mean.index,
        'Mean': metrics_mean.values,
        'Std': metrics_std.values
    })

    # Plot metrics with error bars
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Mean', data=metrics_summary_df, capsize=0.2, palette='viridis', hue='Metric')
    plt.ylabel('Score')
    ax.errorbar(x=metrics_summary_df['Metric'], y=metrics_summary_df['Mean'], yerr=metrics_summary_df['Std'], fmt=' ', c='k')
    plt.title(f'{model_name} Performance Metrics with Standard Deviation')
    plt.show()

    # Create ROC Curve
    y_prob = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')
    fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.lineplot(x=fpr, y=tpr, label=f'{model_name} ROC (area = {metrics_mean["ROC AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')

    # Create Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_prob[:, 1])
    plt.subplot(1, 3, 2)
    sns.lineplot(x=recall, y=precision, label=f'{model_name} Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Create Confusion Matrix Plot
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # Create a lift curve
    skplt.metrics.plot_lift_curve(y, y_prob)
    plt.title(f'{model_name} Lift Curve')
    plt.show()

    return metrics_summary_df