# main.py
import argparse
import sys
from data_loader import load_real_telecom_data
from processing import execute_cleaning_and_quality_logs, engineer_features
from train import execute_model_training_pipeline
from evaluate import run_performance_audit, compute_explainable_ai_layer

def main():
    parser = argparse.ArgumentParser(description="Moroccan Telecom Churn Production ML Pipeline")
    
    # Strict positional/named parameters for your explicit files
    parser.add_argument('--train_path', type=str, required=True, help='Path to your train.csv dataset')
    parser.add_argument('--test_path', type=str, required=True, help='Path to your test.csv dataset')
    parser.add_argument('--threshold', type=float, default=0.40, help='Prediction threshold for Recall')
    parser.add_argument('--explain_row', type=int, default=0, help='Index row for SHAP trace')

    args = parser.parse_args()

    print("=" * 60)
    print("      LAUNCHING AUTOMATED EXPLICIT SPLIT CHURN PIPELINE")
    print("=" * 60)

    try:
        # 1. Load and process Training Dataset
        print(f"\n[1/5] Processing Training Data File: {args.train_path}...")
        raw_train = load_real_telecom_data(args.train_path)
        cleaned_train = execute_cleaning_and_quality_logs(raw_train)
        X_train, y_train = engineer_features(cleaned_train)

        # 2. Load and process Testing Dataset
        print(f"\n[2/5] Processing Testing Data File: {args.test_path}...")
        raw_test = load_real_telecom_data(args.test_path)
        cleaned_test = execute_cleaning_and_quality_logs(raw_test)
        X_test, y_test = engineer_features(cleaned_test)

        # 3. Train models across the explicit sets
        print("\n[3/5] Initializing algorithmic training matrix...")
        lr_m, xgb_m, lgb_m, cat_m = execute_model_training_pipeline(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        # 4. Run Side-By-Side Performance Audits and Save Presentations Visuals
        print("\n[4/5] Computing benchmarks and saving comparison graphics...")
        run_performance_audit(lr_m, xgb_m, lgb_m, cat_m, X_test, y_test, target_threshold=args.threshold)

        # 5. Build Explainable AI maps with your proven winner (CatBoost)
        print("\n[5/5] Extracting champion model insights via SHAP...")
        compute_explainable_ai_layer(cat_m, X_test, customer_idx=args.explain_row)
        
        print("\nPipeline run executed successfully using true training splits!")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline halted: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()