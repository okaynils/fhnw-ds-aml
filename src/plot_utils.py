import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, auc, ConfusionMatrixDisplay)

def get_top_p_percent_predictions(estimator, X, p=0.1):
    probabilities = estimator.predict_proba(X)[:, 1]
    sorted_indices = probabilities.argsort()
    top_indices = sorted_indices[-int(p * len(X)):]
    return X.iloc[top_indices]

def plot_model_concordance(estimators, estimator_names, X, p=0.1):
    overlap_matrix = np.zeros((len(estimators), len(estimators)))

    for i, estimator1 in enumerate(estimators):
        for j, estimator2 in enumerate(estimators):
            if i >= j:
                top_p_percent1 = get_top_p_percent_predictions(estimator1, X, p)
                top_p_percent2 = get_top_p_percent_predictions(estimator2, X, p)
                
                overlap = top_p_percent1.index.intersection(top_p_percent2.index)
                overlap_matrix[i, j] = len(overlap) / len(top_p_percent1)
    
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('coolwarm')
    cmap.set_under('white', alpha=0)
    
    masked_array = np.ma.masked_where(overlap_matrix == 0, overlap_matrix)

    cax = ax.matshow(masked_array, cmap=cmap, vmin=0.01, zorder=2)

    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(estimator_names)))
    ax.set_yticks(np.arange(len(estimator_names)))
    ax.set_xticklabels(estimator_names)
    ax.set_yticklabels(estimator_names)

    plt.title(f'Overlap Matrix for Top {p*100}% Customers')
    
    ax.set_xticklabels(estimator_names, rotation=90)
    ax.set_xticks(np.arange(len(estimator_names)+1)-0.5, minor=True)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.set_yticks(np.arange(len(estimator_names)+1)-0.5, minor=True)
    ax.grid(True, zorder=1, color='black')

    for i in range(len(estimator_names)):
        for j in range(len(estimator_names)):
            if i >= j:
                text = ax.text(j, i, f'{overlap_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    plt.show()

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


def plot_model_evaluation_summary(roc_curves, metrics_mean, best_estimator, X_test, y_test, model_name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    for fpr, tpr in roc_curves:
        ax[0].plot(fpr, tpr, alpha=0.3)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
    ax[0].plot(mean_fpr, mean_tpr, label=f'{model_name} Mean ROC (area = {metrics_mean["ROC AUC"]:.2f})', color='b')
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver Operating Characteristic')
    ax[0].legend(loc='lower right')
    
    y_pred = best_estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')
    ax[1].set_title(f'{model_name} Confusion Matrix')
    
    plt.suptitle(f'Model Evaluation Summary for {model_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    

def plot_auc_boxplots(roc_curves_list, model_name_list):
    auc_values = []
    
    for roc_curves in roc_curves_list:
        model_auc_values = [auc(fpr, tpr) for fpr, tpr in roc_curves]
        auc_values.append(model_auc_values)
    
    medianprops = dict(linestyle='-', linewidth=2, color='k')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(auc_values, vert=True, patch_artist=True, tick_labels=model_name_list, medianprops=medianprops)
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC Value')
    ax.set_title('AUC Boxplots for Multiple Estimators')
    plt.show()
    

def plot_param_grid_heatmap(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)

    param_columns = [col for col in results.columns if col.startswith('param_')]
    if len(param_columns) != 2:
        raise ValueError("The parameter grid must contain exactly two parameters.")

    param1 = param_columns[0]
    param2 = param_columns[1]

    pivot_table = results.pivot(index=param1, columns=param2, values='mean_test_score')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".3f")
    plt.suptitle('Grid Search Scores')
    plt.title(f'AUC Scores for {param1} and {param2}')
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.show()


def plot_multiple_roc_curves(roc_curves_list, model_name_list):
    num_models = len(roc_curves_list)
    num_cols = 2
    num_rows = (num_models + 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 6))
    
    for i, (roc_curves, model_name) in enumerate(zip(roc_curves_list, model_name_list)):
        ax = axes[i // num_cols, i % num_cols]
        all_tpr = []
        for fpr, tpr in roc_curves:
            ax.plot(fpr, tpr, alpha=0.3)
            all_tpr.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
        
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(all_tpr, axis=0)
        std_tpr = np.std(all_tpr, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        roc_auc = np.trapz(mean_tpr, mean_fpr)
        mean_std = np.mean(std_tpr)
        
        ax.plot(mean_fpr, mean_tpr, label=f'{model_name} Mean ROC (area = {roc_auc:.4f})', color='b')
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label=f'{model_name} ± 1 std. dev. (μσ = {mean_std:.4f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} ROC Curve')
        ax.legend(loc='lower right')
    
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])
    
    plt.tight_layout()
    plt.suptitle('Receiver Operating Characteristic Curves for Multiple Estimators', y=1.02)
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


def plot_confusion_matrices(estimators, X_test, y_test, model_names):
    n_estimators = len(estimators)
    nrows = (n_estimators + 1) // 3
    ncols = 3 if n_estimators > 1 else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    axes = axes.flatten() if n_estimators > 1 else [axes]
    
    for i, (estimator, model_name) in enumerate(zip(estimators, model_names)):
        y_pred = estimator.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Card', 'Card'])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d', colorbar=False)
        axes[i].set_title(f'{model_name} CM')
        axes[i].grid(False)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
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