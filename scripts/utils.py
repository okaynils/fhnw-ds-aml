import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from matplotlib.ticker import FuncFormatter
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
    sns.set(style="whitegrid")

    if not model_name:
        model_name = pipeline.steps[-1][1].__class__.__name__ if pipeline.steps else "Estimator"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics_list = []
    roc_curves = []
    pr_curves = []
    lift_probs = []
    true_labels = []

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

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        roc_curves.append((fpr, tpr))
        pr_curves.append((precision, recall))
        lift_probs.append(y_prob)
        true_labels.append(y_test)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_mean = metrics_df.mean()
    metrics_std = metrics_df.std()

    metrics_summary_df = pd.DataFrame({
        'Metric': metrics_mean.index,
        'Mean': metrics_mean.values,
        'Std': metrics_std.values
    })

    # Metrics barplot with error bars
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Mean', data=metrics_summary_df, capsize=0.2, palette='viridis', hue='Metric')
    plt.ylabel('Score')
    ax.errorbar(x=metrics_summary_df['Metric'], y=metrics_summary_df['Mean'], yerr=metrics_summary_df['Std'], fmt=' ', c='k')
    plt.title(f'{model_name} Performance Metrics with Standard Deviation')
    plt.show()

    # ROC Curve
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    for fpr, tpr in roc_curves:
        plt.plot(fpr, tpr, alpha=0.3)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
    plt.plot(mean_fpr, mean_tpr, label=f'{model_name} Mean ROC (area = {metrics_mean["ROC AUC"]:.2f})', color='b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Precision-Recall Curve
    plt.subplot(1, 3, 2)
    for precision, recall in pr_curves:
        plt.plot(recall, precision, alpha=0.3)
    mean_precision = np.linspace(0, 1, 100)
    mean_recall = np.mean([np.interp(mean_precision, recall[::-1], precision[::-1]) for precision, recall in pr_curves], axis=0)
    plt.plot(mean_precision, mean_recall, label=f'{model_name} Mean PR', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    # Confusion Matrix Plot
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # Lift curve
    y_prob = np.concatenate(lift_probs)
    y_true = np.concatenate(true_labels)
    skplt.metrics.plot_lift_curve(y_true, np.vstack([1 - y_prob, y_prob]).T)
    plt.title(f'{model_name} Lift Curve')
    plt.show()

    return metrics_summary_df


def large_number_formatter(x, pos):
    if x >= 1e6:
        return f'{x / 1e6:.1f}M'
    elif x >= 1e3:
        return f'{x / 1e3:.1f}K'
    else:
        return f'{x:.0f}'


def plot_agg_variables(client_df, metric_prefix, aggfuncs=['mean']):
    columns = [f'{metric_prefix}_month_diff_{i}' for i in range(1, 14)]
    
    num_plots = len(aggfuncs)
    
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5 * num_rows))
    
    axes = axes.flatten()
    
    for ax, aggfunc in zip(axes, aggfuncs):
        summary = client_df.groupby('has_card')[columns].agg(aggfunc).T
        
        summary.columns = ['No Card', 'Has Card']
        summary.index = range(1, 14)
        
        sns.lineplot(data=summary, markers=["o", "o"], dashes=False, ax=ax)
        ax.set_xlabel('Month')  
        ax.set_ylabel(f'{aggfunc.capitalize()} {metric_prefix.capitalize()}')
        ax.set_title(f'{aggfunc.capitalize()} {metric_prefix.capitalize()} Over Time by Card Ownership')
        ax.legend(title='Card Ownership')
        ax.grid(True)
        ax.set_xticks(range(1, 14))
        
        ax.yaxis.set_major_formatter(FuncFormatter(large_number_formatter))
    
    for i in range(len(aggfuncs), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()