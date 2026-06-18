# telecom_churn/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, roc_curve, precision_recall_fscore_support
import shap
import matplotlib.pyplot as plt

def run_performance_audit(lr_model, xgb_model, lgb_model, cat_model, X_test, y_test, target_threshold=0.40):
    """
    Generates comparative benchmarks, plots ROC/PR curves, and automatically
    exports text reports and performance images for presentations.
    """
    print("\n" + "="*60 + "\n      ENTERPRISE BENCHMARK PERFORMANCE AUDIT\n" + "="*60)
    
    models = {
        'Logistic Regression': lr_model,
        'Optimized XGBoost': xgb_model,
        'Optimized LightGBM': lgb_model,
        'Optimized CatBoost': cat_model
    }
    
    # Dictionary to hold structured metrics for our exported tables/images
    report_data = []
    
    # Initialize the curves canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = {
        'Logistic Regression': '#7f8c8d',
        'Optimized XGBoost': '#e67e22',
        'Optimized LightGBM': '#2980b9',
        'Optimized CatBoost': '#27ae60'
    }
    
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= target_threshold).astype(int)
        
        # Global Matrix Calculations
        roc_auc = roc_auc_score(y_test, probs)
        precision_labs, recall_labs, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall_labs, precision_labs)
        
        # Calculate Threshold Specific Matrix (Class 1 Churn)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label=1)
        acc = (preds == y_test).mean()
        
        # Append to our structural dictionary list
        report_data.append({
            'Model': name,
            'ROC-AUC': round(roc_auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'Accuracy': round(acc, 4),
            'Precision (Churn)': round(prec, 4),
            'Recall (Churn)': round(rec, 4),
            'F1-Score (Churn)': round(f1, 4)
        })
        
        # Plot Curves
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', color=colors[name], lw=2.5)
        ax2.plot(recall_labs, precision_labs, label=f'{name} (PR-AUC = {pr_auc:.4f})', color=colors[name], lw=2.5)
        
        print(f"{name:<20} -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        
    print("="*60)
    
    # ----------------------------------------------------
    # EXPORT ARTIFACT 1: Markdown Table File
    # ----------------------------------------------------
    metrics_df = pd.DataFrame(report_data)
    markdown_filename = "model_performance_report.md"
    
    with open(markdown_filename, "w") as f:
        f.write("# Moroccan Telecom Churn Prediction Pipeline Run\n\n")
        f.write(f"### Operational Decision Threshold Applied: **{target_threshold}**\n\n")
        f.write(metrics_df.to_markdown(index=False))
    print(f"[SUCCESS] Markdown comparison table saved to: '{markdown_filename}'")
    
    # ----------------------------------------------------
    # EXPORT ARTIFACT 2: Comparative Dashboard Image
    # ----------------------------------------------------
    # We isolate our tree-boosting algorithms for a focused operational bar-chart
    boosting_metrics = metrics_df[metrics_df['Model'].str.contains('XGBoost|LightGBM|CatBoost')]
    
    fig_bars, ax_bars = plt.subplots(figsize=(12, 6))
    x_indices = np.arange(len(boosting_metrics['Model']))
    bar_width = 0.18
    
    # Plot grouped metrics bars
    ax_bars.bar(x_indices - 1.5*bar_width, boosting_metrics['Precision (Churn)'], bar_width, label='Precision (Alert Accuracy)', color='#34495e')
    ax_bars.bar(x_indices - 0.5*bar_width, boosting_metrics['Recall (Churn)'], bar_width, label='Recall (% Caught Churners)', color='#e74c3c')
    ax_bars.bar(x_indices + 0.5*bar_width, boosting_metrics['F1-Score (Churn)'], bar_width, label='F1-Score (Balance)', color='#f1c40f')
    ax_bars.bar(x_indices + 1.5*bar_width, boosting_metrics['Accuracy'], bar_width, label='Overall Accuracy', color='#95a5a6')
    
    # Style the chart bars
    ax_bars.set_title(f'Operational Performance Comparison Dashboard (Threshold: {target_threshold})', fontsize=14, fontweight='bold', pad=15)
    ax_bars.set_xticks(x_indices)
    ax_bars.set_xticklabels(boosting_metrics['Model'], fontsize=11, fontweight='bold')
    ax_bars.set_ylabel('Performance Score Scale (0.0 - 1.0)', fontsize=12)
    ax_bars.set_ylim(0.75, 1.02) # Zooms in on high-performance variations
    ax_bars.legend(loc='lower left', frameon=True, shadow=False)
    ax_bars.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Add clear text value labels on top of each bar
    for patch in ax_bars.patches:
        height = patch.get_height()
        if height > 0:
            ax_bars.annotate(f'{height:.3f}',
                             xy=(patch.get_x() + patch.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    plt.tight_layout()
    bar_filename = "model_metrics_comparison.png"
    plt.savefig(bar_filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Operational metric comparison dashboard saved to: '{bar_filename}'")
    plt.close()

    # Style and Save standard ROC/PR Curves Canvas
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    ax2.set_title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.savefig("model_comparison_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def compute_explainable_ai_layer(best_model, X_test, customer_idx=0):
    """
    Computes global feature importances using the champion model's structures.
    """
    print("\n[SHAP INTEGRATION] Initializing TreeExplainer using champion architecture...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Champion Model Global Feature Importance (SHAP Impact Spectrum)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    shap_filename = "champion_feature_importance.png"
    plt.savefig(shap_filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] SHAP explainability graph saved to: '{shap_filename}'\n")
    plt.close()