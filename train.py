# train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
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
    Trains and tunes boosting models using isolated data frames.
    """
    print("\n" + "="*50 + "\n[TRAINING PIPELINE INITIALIZATION]\n" + "="*50)
    
    # Guarantee test shape matches train perfectly
    if not X_test.empty:
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
    print(f"Dataset Verified -> Training Samples: {X_train.shape[0]} | Test Validation Samples: {X_test.shape[0]}")
    
    # Address class imbalances
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Model 1: Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Model 2: XGBoost
    print("Optimizing XGBoost...")
    xgb_base = xgb.XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
    xgb_param = {'max_depth': [3, 5, 6], 'learning_rate': [0.05, 0.1], 'n_estimators': [50, 100]}
    xgb_search = RandomizedSearchCV(xgb_base, param_distributions=xgb_param, n_iter=2, scoring='roc_auc', cv=cv_strategy, random_state=42, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    
    # Model 3: LightGBM
    print("Optimizing LightGBM...")
    lgb_base = lgb.LGBMClassifier(scale_pos_weight=scale_weight, random_state=42, verbose=-1)
    lgb_param = {'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1], 'n_estimators': [50, 100]}
    lgb_search = RandomizedSearchCV(lgb_base, param_distributions=lgb_param, n_iter=2, scoring='roc_auc', cv=cv_strategy, random_state=42, n_jobs=-1)
    lgb_search.fit(X_train, y_train)
    best_lgb = lgb_search.best_estimator_
    
    # Model 4: CatBoost Champion
    print("Optimizing CatBoost...")
    cat_base = CatBoostClassifier(scale_pos_weight=scale_weight, random_state=42, verbose=0)
    cat_param = {'depth': [4, 6], 'learning_rate': [0.05, 0.1], 'iterations': [100]}
    cat_search = RandomizedSearchCV(cat_base, param_distributions=cat_param, n_iter=2, scoring='roc_auc', cv=cv_strategy, random_state=42, n_jobs=-1)
    cat_search.fit(X_train, y_train)
    best_cat = cat_search.best_estimator_
    
    return lr_model, best_xgb, best_lgb, best_cat