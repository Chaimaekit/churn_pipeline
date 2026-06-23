import sys
import os
import pandas as pd
from data_loader import load_local_training_data
from processing import execute_cleaning_and_quality_logs, engineer_features
from train import execute_model_training_pipeline
# FIX: Removed the non-existent compute_explainable_ai_layer import
from evaluate import run_performance_audit 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
from contextlib import asynccontextmanager
from database.supabase_client import SupabaseDB
from dotenv import load_dotenv

# Automatically pull values from your local .env configuration 
load_dotenv()

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
    """ Handles API startup tasks and instantiates local weights safely in cloud contexts """
    global CHAMPION_MODEL, FEATURE_COLUMNS
    model_path = os.getenv("MODEL_CACHE_PATH", "models/champion_catboost.pkl")
    feature_path = os.getenv("FEATURE_CACHE_PATH", "models/feature_columns.pkl")
    
    if os.path.exists(model_path):
        print(f"[STARTUP] Ingesting cached production weights from: {model_path}")
        CHAMPION_MODEL = joblib.load(model_path)
    else:
        print(f"[STARTUP WARNING] No active model binaries found. Endpoints require hitting /run_pipeline first.")
        
    if os.path.exists(feature_path):
        FEATURE_COLUMNS = joblib.load(feature_path)
        
    yield 
    print("[SHUTDOWN] Tearing down runtime engine connections...")


app = FastAPI(lifespan=lifespan)
db = SupabaseDB()

def preprocess_single_customer(raw_data: dict) -> pd.DataFrame:
    """ Transforms a single record into the structure expected by the model """
    df_single = pd.DataFrame([raw_data])
    cleaned_df = execute_cleaning_and_quality_logs(df_single)
    X_single, _ = engineer_features(cleaned_df)
    
    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is not None:
        X_single = X_single.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return X_single

def churn_pipeline(train_path: str, target_threshold=0.40, explain_row=0):
    """ Trains models using local historical files and tests against clean cloud rows """
    global CHAMPION_MODEL, FEATURE_COLUMNS
    print("\n" + "="*60 + "\n      STARTING SUPABASE-INTEGRATED PIPELINE RUN\n" + "="*60)
    
    # 1. Load Local Historical Data
    print(f"[1/5] Ingesting Local Historical Training Parameters from: {train_path}")
    raw_train = load_local_training_data(train_path)
    cleaned_train = execute_cleaning_and_quality_logs(raw_train)
    X_train, y_train = engineer_features(cleaned_train)
    
    # Save training feature map matrix configuration
    os.makedirs("models", exist_ok=True)
    FEATURE_COLUMNS = X_train.columns.tolist()
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.pkl")

    # 2. Extract Evaluation Sets Straight from Cloud Tables
    print("\n[2/5] Fetching production test evaluation frames from Supabase 'subscribers' table...")
    if db.client is not None:
        res = db.client.table("subscribers").select("*").execute()
        if not res.data:
            raise ValueError("Supabase 'subscribers' table contains no records for pipeline verification.")
        raw_test = pd.DataFrame(res.data)
    else:
        raise ConnectionError("Supabase client interface is uninitialized. Verify your environment keys.")

    cleaned_test = execute_cleaning_and_quality_logs(raw_test)
    X_test, y_test = engineer_features(cleaned_test)
    X_test = X_test.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # 3. Fit Model Pipelines
    print("\n[3/5] Trait-fitting models clean across isolated datasets...")
    lr_m, xgb_m, lgb_m, cat_m = execute_model_training_pipeline(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # 4. Audit Comparative Benchmarks
    print("\n[4/5] Executing performance audits and exporting monitoring charts...")
    run_performance_audit(lr_m, xgb_m, lgb_m, cat_m, X_test, y_test, target_threshold=target_threshold)

    # 5. Model Calibration Complete
    print("\n[5/5] Finalizing core classifier optimizations...")
    
    # Serialize model weights locally and update cache reference
    joblib.dump(cat_m, "models/champion_catboost.pkl")
    CHAMPION_MODEL = cat_m
    print("\n[SUCCESS] Local model optimized and verified against remote test frames successfully!")

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API cloud processing pipeline is active."}

@app.post("/run_pipeline")
def run_pipeline_endpoint(train_path: str = "data/train_with_feedback.csv", threshold: float = 0.40, explain_row: int = 0):
    try:
        churn_pipeline(train_path, target_threshold=threshold, explain_row=explain_row)
        return {"status": "success", "message": "Model successfully optimized locally and verified against clean Supabase records."}
    except Exception as e:
        return {"status": "error", "message": f"Pipeline engine execution sequence failed: {str(e)}"}

@app.post("/prediction/realtime")
def predict_realtime_churn(payload: CustomerPayload, threshold: float = 0.40):
    """ Runs on-the-fly analytical predictions over volatile client streams and saves them to the cloud """
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Champion model weights not compiled. Call /run_pipeline first.")
    
    try:
        customer_dict = payload.dict()
        X_processed = preprocess_single_customer(customer_dict)
        
        prob = float(CHAMPION_MODEL.predict_proba(X_processed)[0][1])
        risk_level = "High" if prob >= threshold else "Low"
        action = prob >= threshold
        
        pred_payload = {
            "customer_id": payload.customer_id,
            "churn_probability": round(prob, 4),
            "churn_risk_level": risk_level,
            "action_required": action,
            "model_version": "champion_catboost"
        }
        
        if db.client is not None:
            db.client.table("predictions").insert(pred_payload).execute()
            db_sync = "Successfully logged tracking rows to Supabase 'predictions' table."
        else:
            db_sync = "Fallback mode: Client not linked to active database targets."

        return {
            "customer_id": payload.customer_id,
            "churn_probability": round(prob, 4),
            "churn_risk_level": risk_level,
            "action_required": action,
            "database_sync": db_sync
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference processing validation failed: {str(e)}")

@app.get("/prediction/{customer_id}")
def predict_customer_from_db(customer_id: str, threshold: float = 0.40):
    """ Pulls an existing subscriber straight from Supabase to execute inference checks """
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Machine learning models are not compiled or active yet.")
        
    if db.client is None:
        raise HTTPException(status_code=500, detail="Supabase infrastructure connection target missing.")
        
    try:
        target_id = str(customer_id).strip()
        res = db.client.table("subscribers").select("*").eq("customer_id", target_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail=f"Customer ID {target_id} does not exist in Supabase tables.")
            
        mock_profile = res.data[0]
        X_processed = preprocess_single_customer(mock_profile)
        
        prob = float(CHAMPION_MODEL.predict_proba(X_processed)[0][1])
        category = mock_profile.get("feedback_category", "unknown")
        
        # Override rules context to protect risky accounts flag positions
        if prob >= threshold or category == "churn_intent":
            risk_level = "High"
        else:
            risk_level = "Low"
            
        return {
            "customer_id": target_id,
            "state": mock_profile.get("state", "unknown"),
            "raw_text_analyzed": mock_profile.get("feedback_text", ""),
            "extracted_category": category,
            "extracted_sentiment": mock_profile.get("sentiment", "unknown"),
            "ml_churn_probability": round(prob, 4),
            "final_churn_risk_level": risk_level
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference engine extraction sequence failed: {str(e)}")

@app.get("/reputation")
def calculate_brand_reputation():
    """ Ingests cloud feedback sentiment metrics to return brand health insights """
    if db.client is None:
        raise HTTPException(status_code=500, detail="Supabase connection target is uninitialized.")
        
    try:
        res = db.client.table("processed_feedback").select("*").execute()
        if not res.data:
            return {"message": "The cloud processed analytic feedback table is currently empty."}
            
        df = pd.DataFrame(res.data)
        total_comments = len(df)
        
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
            "monitored_source": "Supabase 'processed_feedback' Table",
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
        raise HTTPException(status_code=500, detail=f"Failed to extract brand reputation statistics: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)