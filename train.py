# telecom_churn/train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from typing import Tuple, Any

def execute_model_training_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Any, Any, Any, Any]:
    """
    Trains and tunes boosting models using your explicitly separated 
    train and test files.
    """
    print("\n" + "="*50 + "\n[TRAINING PIPELINE INITIALIZATION]\n" + "="*50)
    
    # Force column realignment to guarantee test matches train perfectly
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
    print(f"Dataset Verified -> Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    
    # Compute Class Imbalance Weights based on the Training file distribution
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Computed scale_pos_weight balance parameter: {scale_weight:.3f}")
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # ----------------------------------------------------
    # MODEL 1: Baseline Logistic Regression
    # ----------------------------------------------------
    print("Fitting Baseline Logistic Regression...")
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    
    # ----------------------------------------------------
    # MODEL 2: Optimized XGBoost
    # ----------------------------------------------------
    print("Optimizing XGBoost Hyperparameters...")
    xgb_base = xgb.XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
    xgb_param_dist = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    xgb_search = RandomizedSearchCV(xgb_base, param_distributions=xgb_param_dist, n_iter=3,
                                    scoring='roc_auc', cv=cv_strategy, random_state=42, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    
    # ----------------------------------------------------
    # MODEL 3: Optimized LightGBM
    # ----------------------------------------------------
    print("Optimizing LightGBM Hyperparameters...")
    lgb_base = lgb.LGBMClassifier(scale_pos_weight=scale_weight, random_state=42, verbose=-1)
    lgb_param_dist = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    lgb_search = RandomizedSearchCV(lgb_base, param_distributions=lgb_param_dist, n_iter=3,
                                    scoring='roc_auc', cv=cv_strategy, random_state=42, n_jobs=-1)
    lgb_search.fit(X_train, y_train)
    best_lgb = lgb_search.best_estimator_
    
    # ----------------------------------------------------
    # MODEL 4: Optimized CatBoost
    # ----------------------------------------------------
    print("Optimizing CatBoost Hyperparameters...")
    cat_base = CatBoostClassifier(scale_pos_weight=scale_weight, random_state=42, verbose=0)
    cat_param_dist = {
        'depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'iterations': [100, 200]
    }
    cat_search = RandomizedSearchCV(cat_base, param_distributions=cat_param_dist, n_iter=3,
                                    scoring='roc_auc', cv=cv_strategy, random_state=42, n_jobs=-1)
    cat_search.fit(X_train, y_train)
    best_cat = cat_search.best_estimator_
    
    return lr_pipe, best_xgb, best_lgb, best_cat