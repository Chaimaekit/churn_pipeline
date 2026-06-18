import sys
import os
import pandas as pd
from data_loader import load_real_telecom_data
from processing import execute_cleaning_and_quality_logs, engineer_features
from train import execute_model_training_pipeline
from evaluate import run_performance_audit, compute_explainable_ai_layer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
from contextlib import asynccontextmanager



test_data={
    "customer_id": "CUST_002667",
    "state": "LA",
    "account_length": 117,
    "area_code": 408,
    "international_plan": "No",
    "voice_mail_plan": "No",
    "number_vmail_messages": 0,
    "total_day_minutes": 184.5,
    "total_day_calls": 97,
    "total_day_charge": 31.37,
    "total_eve_minutes": 351.6,
    "total_eve_calls": 80,
    "total_eve_charge": 29.89,
    "total_night_minutes": 215.8,
    "total_night_calls": 90,
    "total_night_charge": 9.71,
    "total_intl_minutes": 8.7,
    "total_intl_calls": 4,
    "total_intl_charge": 2.35,
    "customer_service_calls": 1,
    "split": 0,
    "has_international_plan": 0,
    "has_voice_mail_plan": 0,
    "total_minutes": 760.6000000000001,
    "total_calls": 271,
    "total_charges": 73.32,
    "avg_charge_per_minute": 0.0963975808572179,
    "support_call_rate": 0.0085470085470085,
    "high_service_calls": 0,
    "usage_intensity": "high_usage",
    "customer_value_segment": "high_value",
    "rule_based_churn_risk_score": 25,
    "rule_based_churn_risk_level": "low",
    "feedback_text": "I use the service a lot, but the monthly cost is becoming high.",
    "feedback_category": "pricing",
    "sentiment": "neutral",
    "complaint_intensity": 3
}


CHAMPION_MODEL = None
FEATURE_COLUMNS = None

class CustomerPayload(BaseModel):
    customer_id: str
    state: str
    account_length: int
    area_code: int
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    customer_service_calls: int
    feedback_text: str = ""
    complaint_intensity: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles API startup and shutdown tasks seamlessly.
    """
    global CHAMPION_MODEL, FEATURE_COLUMNS
    model_path = "models/champion_catboost.pkl"
    
    # --- Startup Logic ---
    if os.path.exists(model_path):
        print(f"[STARTUP] Loading saved champion model from {model_path}...")
        CHAMPION_MODEL = joblib.load(model_path)
    else:
        print(f"[STARTUP WARNING] No saved model found at {model_path}. Please run the pipeline first.")
        
    yield 
    print("[SHUTDOWN] Cleaning up resources...")


app = FastAPI(lifespan= lifespan)


def preprocess_single_customer(raw_data: dict) -> pd.DataFrame:
    """
    Transforms a single raw customer record into the exact DataFrame structure 
    expected by your engineered model matrix.
    """
    df_single = pd.DataFrame([raw_data])
    
    cleaned_df = execute_cleaning_and_quality_logs(df_single)
    X_single, _ = engineer_features(cleaned_df)
    
    if FEATURE_COLUMNS is not None:
        for col in FEATURE_COLUMNS:
            if col not in X_single.columns:
                X_single[col] = 0
        X_single = X_single[FEATURE_COLUMNS]
        
    return X_single

@app.post("/prediction/realtime")
def predict_realtime_churn(payload: CustomerPayload):
    """Predicts churn risk for an on-the-fly JSON payload sent directly over HTTP."""
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or trained yet.")
    
    try:
        customer_dict = payload.dict()
        X_processed = preprocess_single_customer(customer_dict)
        
        # Calculate real-time churn metrics
        probability = float(CHAMPION_MODEL.predict_proba(X_processed)[0][1])
        prediction = int(probability >= 0.40) # matching your target_threshold
        
        return {
            "customer_id": payload.customer_id,
            "churn_probability": round(probability, 4),
            "churn_risk_level": "High" if probability >= 0.40 else "Low",
            "action_required": prediction == 1
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference processing failed: {str(e)}")


@app.get("/prediction/{customer_id}")
def predict_customer_from_batch(customer_id: str):
    """
    Looks up a structural ID from the processed LLM feedback CSV
    and applies machine learning prediction rules to calculate churn probability.
    """
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Machine Learning model weights are not loaded or trained yet.")
        
    batch_file_path = "data/processed_feedback.csv"
    
    if not os.path.exists(batch_file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Processed file not found at '{batch_file_path}'. Please execute 'feedback_analyse.py' first."
        )
        
    try:
        raw_batch = pd.read_csv(batch_file_path)
        
        # Clean up trailing float extensions from the path string
        target_id = str(customer_id).strip()
        if target_id.endswith(".0"):
            target_id = target_id[:-2]
            
        # Clean up existing dataframe IDs so string comparisons evaluate cleanly
        raw_batch['customer_id'] = raw_batch['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # Look up match
        customer_row = raw_batch[raw_batch['customer_id'] == target_id]
        if customer_row.empty:
            raise HTTPException(status_code=404, detail=f"Scraped ID {target_id} not found in the processed LLM table.")
            
        customer_dict = customer_row.iloc[0].to_dict()
        
        # FIX: Create a shallow copy of the baseline template to prevent mutating test_data globally
        mock_profile = test_data.copy()
        
        # Inject the dynamically extracted data from your LLM processed scrape
        mock_profile["customer_id"] = target_id
        mock_profile["feedback_text"] = customer_dict.get("raw_text", "")
        mock_profile["complaint_intensity"] = int(customer_dict.get("complaint_intensity", 1))
        mock_profile["feedback_category"] = customer_dict.get("feedback_category", "unknown")
        mock_profile["sentiment"] = customer_dict.get("sentiment", "unknown")
        
        # Overwrite your model's engineered proxy feature directly based on LLM scores
        mock_profile["customer_service_calls"] = mock_profile["complaint_intensity"]
        
        # Explicit target placeholder layer bypass logic
        mock_profile["churn_flag"] = 0
        
        # Process the profile through your original feature pipelines
        X_processed = preprocess_single_customer(mock_profile)
        
        # Calculate real prediction array metrics
        probability = float(CHAMPION_MODEL.predict_proba(X_processed)[0][1])
        
        return {
            "scraped_id": target_id,
            "raw_text_analyzed": mock_profile["feedback_text"],
            "llm_extracted_category": mock_profile["feedback_category"],
            "llm_extracted_sentiment": mock_profile["sentiment"],
            "ml_churn_probability": round(probability, 4),
            "final_churn_risk_level": "High" if probability >= 0.40 or mock_profile["feedback_category"] == "churn_intent" else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference engine extraction sequence failed: {str(e)}")

# Keep your existing pipeline runner logic intact...
def churn_pipeline(train_path, test_path, target_threshold=0.40, explain_row=0):
    print("        LAUNCHING AUTOMATED EXPLICIT SPLIT CHURN PIPELINE")
    print("=" * 60)
    try:
        print(f"\n[1/5] Processing Training Data File: {train_path}...")
        raw_train = load_real_telecom_data(train_path)
        cleaned_train = execute_cleaning_and_quality_logs(raw_train)
        X_train, y_train = engineer_features(cleaned_train)

        print(f"\n[2/5] Processing Testing Data File: {test_path}...")
        raw_test = load_real_telecom_data(test_path)
        cleaned_test = execute_cleaning_and_quality_logs(raw_test)
        X_test, y_test = engineer_features(cleaned_test)

        print("\n[3/5] Initializing algorithmic training matrix...")
        lr_m, xgb_m, lgb_m, cat_m = execute_model_training_pipeline(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        print("\n[4/5] Computing benchmarks and saving comparison graphics...")
        run_performance_audit(lr_m, xgb_m, lgb_m, cat_m, X_test, y_test, target_threshold=target_threshold)

        print("\n[5/5] Extracting champion model insights via SHAP...")
        compute_explainable_ai_layer(cat_m, X_test, customer_idx=explain_row)
        
        # --- NEW: Serialize the champion model so our API can load it on boot ---
        os.makedirs("models", exist_ok=True)
        joblib.dump(cat_m, "models/champion_catboost.pkl")
        # joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")
        global CHAMPION_MODEL
        CHAMPION_MODEL = cat_m
        
        print("\nPipeline run executed successfully & Champion Model Serialized!")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline halted: {str(e)}", file=sys.stderr)
        sys.exit(1)

@app.post("/run_pipeline")
def run_pipeline_endpoint(train_path: str="data/train_with_feedback.csv", test_path: str="data/test_with_feedback.csv", threshold: float = 0.40, explain_row: int = 0):
    try:
        churn_pipeline(train_path, test_path, target_threshold=threshold, explain_row=explain_row)
        return {"status": "success", "message": "Pipeline executed and model artifacts updated successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Pipeline execution failed: {str(e)}"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is up and running!"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)