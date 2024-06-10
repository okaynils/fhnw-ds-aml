import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

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


def plot_learning_curve(estimator, X, y, n_folds=5, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=n_folds,
        n_jobs=n_jobs
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + train_std, test_mean - train_std, alpha=0.15, color='green')

    plt.title('Learning Curve')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
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