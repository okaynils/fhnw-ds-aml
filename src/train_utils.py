import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.feature_selection import RFE


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
            ('num', StandardScaler(), num_columns.columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns.columns)
        ])
    
    return preprocessor, column_selection_train


class SelectFromModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.rfe = None

    def fit(self, X, y=None):
        self.rfe = RFE(estimator=self.estimator, n_features_to_select=self.n_features_to_select)
        self.rfe.fit(X, y)
        return self

    def transform(self, X):
        return self.rfe.transform(X)


def cross_validate(pipeline, X, y, n_splits=5, param_grid=None, n_features_to_select=None, random_state=1337):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    best_estimator = pipeline
    best_params = None
    
    if param_grid is not None:
        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=-1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
    
    metrics_list = []
    roc_curves = []
    pr_curves = []
    lift_probs = []
    true_labels = []

    for train_index, test_index in tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="Cross-Validation"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if param_grid is not None:
            best_estimator = grid_search.best_estimator_

        if n_features_to_select is not None:
            rfe_step = SelectFromModelTransformer(estimator=best_estimator.named_steps['classifier'], n_features_to_select=n_features_to_select)
            pipeline_with_rfe = Pipeline([
                ('preprocessor', best_estimator.named_steps['preprocessor']),
                ('feature_selection', rfe_step),
                ('classifier', best_estimator.named_steps['classifier'])
            ])
            best_estimator = pipeline_with_rfe

        if hasattr(best_estimator.named_steps['classifier'], 'random_state'):
            best_estimator.named_steps['classifier'].random_state = random_state

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