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
    "account_length": 49,
    "area_code": 510,
    "international_plan": "No",
    "voice_mail_plan": "No",
    "number_vmail_messages": 0,
    "total_day_minutes": 119.3,
    "total_day_calls": 117,
    "total_day_charge": 20.28,
    "total_eve_minutes": 215.1,
    "total_eve_calls": 109,
    "total_eve_charge": 18.28,
    "total_night_minutes": 178.7,
    "total_night_calls": 90,
    "total_night_charge": 8.04,
    "total_intl_minutes": 11.1,
    "total_intl_calls": 1,
    "total_intl_charge": 3.0,
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
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(cat_m, "models/champion_catboost.pkl")
        global CHAMPION_MODEL
        CHAMPION_MODEL = cat_m
        
        print("\nPipeline run executed successfully & Champion Model Serialized!")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline halted: {str(e)}", file=sys.stderr)
        sys.exit(1)


@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is up and running!"}


@app.post("/run_pipeline")
def run_pipeline_endpoint(train_path: str="data/train_with_feedback.csv", test_path: str="data/test_with_feedback.csv", threshold: float = 0.40, explain_row: int = 0):
    try:
        churn_pipeline(train_path, test_path, target_threshold=threshold, explain_row=explain_row)
        return {"status": "success", "message": "Pipeline executed and model artifacts updated successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Pipeline execution failed: {str(e)}"}


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
    Looks up an authentic customer record from 'data/test_with_feedback.csv'
    and applies machine learning prediction rules to evaluate real-time churn metrics.
    """
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Machine Learning model weights are not loaded or trained yet.")
        
    batch_file_path = "data/test_with_feedback.csv"
    
    if not os.path.exists(batch_file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Test batch file not found at '{batch_file_path}'. Please run the pipeline or place the file first."
        )
        
    try:
        raw_batch = pd.read_csv(batch_file_path)
        
        target_id = str(customer_id).strip()
        if target_id.endswith(".0"):
            target_id = target_id[:-2]
            
        raw_batch['customer_id'] = raw_batch['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        customer_row = raw_batch[raw_batch['customer_id'] == target_id]
        if customer_row.empty:
            raise HTTPException(status_code=404, detail=f"Customer ID {target_id} not found inside '{batch_file_path}'.")
            
        mock_profile = customer_row.iloc[0].to_dict()
        
        if 'churn_flag' not in mock_profile and 'churn' in mock_profile:
            mock_profile['churn_flag'] = mock_profile['churn']
        elif 'churn_flag' not in mock_profile:
            mock_profile['churn_flag'] = 0
            
        X_processed = preprocess_single_customer(mock_profile)
        
        probability = float(CHAMPION_MODEL.predict_proba(X_processed)[0][1])
        
        return {
            "customer_id": target_id,
            "state": mock_profile.get("state", "unknown"),
            "raw_text_analyzed": mock_profile.get("feedback_text", ""),
            "extracted_category": mock_profile.get("feedback_category", "unknown"),
            "extracted_sentiment": mock_profile.get("sentiment", "unknown"),
            "ml_churn_probability": round(probability, 4),
            "final_churn_risk_level": "High" if probability >= 0.40 or mock_profile.get("feedback_category") == "churn_intent" else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference engine extraction sequence failed: {str(e)}")


@app.get("/reputation")
def calculate_brand_reputation():
    """
    Ingests the LLM processed feedback summary metrics (Moroccan Darija sentiments) 
    to output high-level KPIs on social brand health, sentiment distributions, and core issues.
    """
    processed_feedback_path = "data/processed_feedback.csv"
    
    if not os.path.exists(processed_feedback_path):
        raise HTTPException(
            status_code=404,
            detail=f"Processed feedback file not found at '{processed_feedback_path}'. Please run 'feedback_analyse.py' first."
        )
        
    try:
        df = pd.read_csv(processed_feedback_path)
        total_comments = len(df)
        
        if total_comments == 0:
            return {"message": "The processed analytics source tracking index is currently empty."}
            
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        category_counts = df['feedback_category'].value_counts().to_dict()
        
        pos = sentiment_counts.get('positive', 0)
        neu = sentiment_counts.get('neutral', 0)
        neg = sentiment_counts.get('negative', 0)
        
        net_reputation_score = round(((pos - neg) / total_comments) * 100, 2)
        
        if net_reputation_score >= 30:
            brand_health = "Excellent / Highly Positive Brand Equity"
        elif net_reputation_score >= 0:
            brand_health = "Stable / Generally Neutral"
        elif net_reputation_score >= -30:
            brand_health = "At Risk / Negative Operational Friction"
        else:
            brand_health = "Critical Alert / High Churn and Public Backlash"
            
        avg_intensity = round(float(df['complaint_intensity'].mean()), 2) if 'complaint_intensity' in df.columns else 0.0
        
        return {
            "monitored_source_file": processed_feedback_path,
            "total_social_records_evaluated": total_comments,
            "net_reputation_score": net_reputation_score,
            "brand_health_status": brand_health,
            "average_complaint_intensity": avg_intensity,
            "sentiment_distribution": {
                "positive": pos,
                "neutral": neu,
                "negative": neg
            },
            "top_operational_complaints": category_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate brand reputation statistics: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)