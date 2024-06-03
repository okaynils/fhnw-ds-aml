import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

def add_prefix_except_id(df, prefix, id_exceptions=[]):
    new_df = df.copy()
    
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
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed['Date'] = X_transformed['Date'].astype('int64') // 10**9
        return X_transformed


def train_test_split_bal(df, target_column, test_size=0.2, random_state=1337, balancing_technique=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if balancing_technique == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def create_pipeline(categorical_features, numerical_features, estimator: BaseEstimator):
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


def build_preprocessor_pipeline(X_train, X_test, include_columns=None, regex_columns=None):
    if include_columns is None:
        include_columns = X_train.columns.tolist()
    
    column_selection_train = X_train[include_columns]
    
    if regex_columns:
        additional_columns = X_train.filter(regex=regex_columns, axis=1).columns
        column_selection_train = pd.concat([column_selection_train, X_train[additional_columns]], axis=1)
    
    cat_columns = column_selection_train.select_dtypes(include=['object'])
    num_columns = column_selection_train.select_dtypes(exclude=['object'])
    
    column_selection_test = X_test[include_columns]
    if regex_columns:
        column_selection_test = pd.concat([column_selection_test, X_test[additional_columns]], axis=1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_columns.columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns.columns)
        ])
    
    return preprocessor, column_selection_train, column_selection_test


def cross_validate(pipeline, X, y, n_splits=5, param_grid=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    
    best_estimator = pipeline
    best_params = None
    
    if param_grid is not None:
        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=-1)
        grid_search.fit(X, y)

        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
    
    metrics_list = []
    roc_curves = []
    pr_curves = []
    lift_probs = []
    true_labels = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        
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
    
    return best_estimator, best_params, metrics_df, roc_curves, pr_curves, lift_probs, true_labels


def plot_metrics(metrics_df, model_name):
    metrics_mean = metrics_df.mean()
    metrics_std = metrics_df.std()
    
    metrics_summary_df = pd.DataFrame({
        'Metric': metrics_mean.index,
        'Mean': metrics_mean.values,
        'Std': metrics_std.values
    })
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Mean', data=metrics_summary_df, capsize=0.2, palette='viridis', hue='Metric')
    plt.ylabel('Score')
    ax.errorbar(x=metrics_summary_df['Metric'], y=metrics_summary_df['Mean'], yerr=metrics_summary_df['Std'], fmt=' ', c='k')
    
    for i in range(len(metrics_summary_df)):
        ax.text(i, metrics_summary_df['Mean'][i] + metrics_summary_df['Std'][i], 
                f"{metrics_summary_df['Mean'][i]:.2f} ± {metrics_summary_df['Std'][i]:.2f}", 
                ha='center', va='bottom')
    
    plt.title(f'{model_name} Performance Metrics with Standard Deviation')
    plt.show()


def plot_roc_curve(roc_curves, metrics_mean, model_name):
    plt.figure(figsize=(6, 6))
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
    plt.suptitle('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


def plot_precision_recall_curve(pr_curves, model_name):
    plt.figure(figsize=(6, 6))
    for precision, recall in pr_curves:
        plt.plot(recall, precision, alpha=0.3)
    mean_precision = np.linspace(0, 1, 100)
    mean_recall = np.mean([np.interp(mean_precision, recall[::-1], precision[::-1]) for precision, recall in pr_curves], axis=0)
    plt.plot(mean_precision, mean_recall, label=f'{model_name} Mean PR', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()


def plot_confusion_matrix(best_estimator, X_test, y_test, model_name):
    y_pred = best_estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()


def plot_lift_curve(lift_probs, true_labels, model_name):
    y_prob = np.concatenate(lift_probs)
    y_true = np.concatenate(true_labels)
    skplt.metrics.plot_lift_curve(y_true, np.vstack([1 - y_prob, y_prob]).T)
    plt.title(f'{model_name} Lift Curve')
    plt.show()


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
