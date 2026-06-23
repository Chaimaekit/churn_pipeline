# evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

def run_performance_audit(lr_model, xgb_model, lgb_model, cat_model, X_test, y_test, target_threshold=0.40):
    """
    Evaluates trained models against the incoming test frames.
    """
    print("\n" + "="*60 + "\n      ENTERPRISE BENCHMARK PERFORMANCE AUDIT\n" + "="*60)
    
    models = {
        'Logistic Regression': lr_model,
        'Optimized XGBoost': xgb_model,
        'Optimized LightGBM': lgb_model,
        'Optimized CatBoost': cat_model
    }
    
    for name, model in models.items():
        preds_proba = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, preds_proba)
        print(f">> {name} ROC-AUC: {auc_val:.4f}")
        
        # Hard predictions based on dynamic threshold criteria
        hard_preds = (preds_proba >= target_threshold).astype(int)
        print(f"Classification Report for {name} (Threshold {target_threshold}):")
        print(classification_report(y_test, hard_preds))
        print("-" * 50)