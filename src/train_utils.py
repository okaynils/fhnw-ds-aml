import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

np.random.seed(1337)

def train_test_split_bal(df, target_column, test_size=0.2, random_state=1337, balancing_technique=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if balancing_technique == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    
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


def build_preprocessor_pipeline(X_train, include_columns=None, regex_columns=None):
    if include_columns is None:
        include_columns = X_train.columns.tolist()
    
    column_selection_train = X_train[include_columns]
    
    if regex_columns:
        additional_columns = X_train.filter(regex=regex_columns, axis=1).columns
        column_selection_train = pd.concat([column_selection_train, X_train[additional_columns]], axis=1)
    
    cat_columns = column_selection_train.select_dtypes(include=['object'])
    num_columns = column_selection_train.select_dtypes(exclude=['object'])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_columns.columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns.columns)
        ])
    
    return preprocessor, column_selection_train


def process_fold(train_index, test_index, X, y, best_estimator, param_grid, grid_search, random_state):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    estimator = best_estimator
    if param_grid is not None:
        estimator = grid_search.best_estimator_

    if hasattr(estimator.named_steps['classifier'], 'random_state'):
        estimator.named_steps['classifier'].random_state = random_state

    estimator.fit(X_train, y_train)
    
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    return metrics, (fpr, tpr), (precision, recall), y_prob, y_test

def cross_validate(pipeline, X, y, n_splits=5, param_grid=None, random_state=1337, n_features_to_select=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    best_estimator = pipeline
    best_params = None
    
    if n_features_to_select is not None:
        classifier = pipeline.named_steps['classifier']
        rfe = RFE(estimator=classifier, n_features_to_select=n_features_to_select, step=10)
        new_pipeline_steps = [(name, step) for name, step in pipeline.steps if name != 'classifier']
        new_pipeline_steps.append(('rfe', rfe))
        new_pipeline_steps.append(('classifier', classifier))
        best_estimator = Pipeline(steps=new_pipeline_steps)
    
    if param_grid is not None:
        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=-1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
    else:
        grid_search = None
    
    results = Parallel(n_jobs=-1)(
        delayed(process_fold)(train_index, test_index, X, y, best_estimator, param_grid, grid_search, random_state) 
        for train_index, test_index in tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="Cross-Validation")
    )
    
    metrics_list, roc_curves, pr_curves, lift_probs, true_labels = zip(*results)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Fit the best estimator on the whole dataset
    if param_grid is not None:
        best_estimator = grid_search.best_estimator_

    if hasattr(best_estimator.named_steps['classifier'], 'random_state'):
        best_estimator.named_steps['classifier'].random_state = random_state

    best_estimator.fit(X, y)

    return best_estimator, best_params, metrics_df, roc_curves, pr_curves, lift_probs, true_labels