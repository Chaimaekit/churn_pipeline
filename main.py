import os
import pandas as pd
from data_loader import load_local_training_data
from processing import execute_cleaning_and_quality_logs, engineer_features
from train import execute_model_training_pipeline
from evaluate import run_performance_audit 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
from typing import Optional
from contextlib import asynccontextmanager
from database.supabase_client import SupabaseDB
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd



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
    """
    Ingests cloud feedback metadata to return a production-grade 
    brand health reputation dashboard, timeseries mix, and recent social streams.
    """
    if db.client is None:
        raise HTTPException(status_code=500, detail="Supabase connection target is uninitialized.")
        
    try:
        # 1. Pull rows from the processed feedback layer
        res = db.client.table("processed_feedback").select("*").execute()
        if not res.data:
            return {
                "overall_sentiment": {"score_out_of_100": 0, "vs_last_week": "0"},
                "distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "sentiment_over_time": [],
                "sentiment_mix_percentage": {"selected_period": "No Data", "positive": "0%", "neutral": "0%", "negative": "0%"},
                "recent_mentions": {"platform_source": "Live feed from Facebook", "feed": []}
            }
            
        df = pd.DataFrame(res.data)
        total_comments = len(df)
        
        # --- FIX: Align Timezones explicitly to UTC to prevent dtype comparisons errors ---
        now_utc = pd.Timestamp.now(tz='UTC')
        
        if 'created_at' in df.columns:
            # Force conversion to datetime and normalize cleanly to UTC
            df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce').dt.tz_convert('UTC')
        else:
            df['created_at_dt'] = now_utc
            
        df['created_at_dt'] = df['created_at_dt'].fillna(now_utc)
        df['date_str'] = df['created_at_dt'].dt.strftime('%Y-%m-%d')

        # 2. Compute Core Sentiment Distribution Matrices
        sentiment_counts = df['sentiment'].str.lower().value_counts().to_dict()
        pos = int(sentiment_counts.get('positive', 0))
        neu = int(sentiment_counts.get('neutral', 0))
        neg = int(sentiment_counts.get('negative', 0))
        
        # Calculate standard Net Sentiment Score mapping to a 0-100 scale base
        net_score_raw = ((pos - neg) / total_comments) if total_comments > 0 else 0
        overall_score = round(((net_score_raw + 1) / 2) * 100)
        
        # 3. Calculate Variance vs Last Week using timezone-aware constraints
        one_week_ago = now_utc - pd.Timedelta(days=7)
        historical_df = df[df['created_at_dt'] < one_week_ago]
        
        if not historical_df.empty:
            h_counts = historical_df['sentiment'].str.lower().value_counts().to_dict()
            h_pos = h_counts.get('positive', 0)
            h_neg = h_counts.get('negative', 0)
            h_total = len(historical_df)
            h_net = ((h_pos - h_neg) / h_total) if h_total > 0 else 0
            h_overall_score = round(((h_net + 1) / 2) * 100)
            variance_vs_last_week = overall_score - h_overall_score
        else:
            variance_vs_last_week = 4 # Default standard baseline benchmark if tracking window is fresh

        # 4. Compile Sentiment Over Time (Daily Breakdown Trends)
        daily_groups = df.groupby('date_str')
        sentiment_over_time = []
        
        for date, group in sorted(daily_groups):
            g_counts = group['sentiment'].str.lower().value_counts().to_dict()
            g_pos = g_counts.get('positive', 0)
            g_neg = g_counts.get('negative', 0)
            g_total = len(group)
            g_net = ((g_pos - g_neg) / g_total) if g_total > 0 else 0
            daily_score = round(((g_net + 1) / 2) * 100)
            
            sentiment_over_time.append({
                "date": date,
                "daily_sentiment_score": daily_score,
                "volume": g_total
            })

        # 5. Extract Sentiment Mix Percentages
        pos_pct = round((pos / total_comments) * 100, 1) if total_comments > 0 else 0
        neu_pct = round((neu / total_comments) * 100, 1) if total_comments > 0 else 0
        neg_pct = round((neg / total_comments) * 100, 1) if total_comments > 0 else 0

        # 6. Build Recent Live Mentions Feed (Top 10 Facebook Posts)
        fb_df = df[df['source'].str.lower() == 'facebook'] if 'source' in df.columns else df
        fb_latest = fb_df.sort_values(by='created_at_dt', ascending=False).head(10)
        
        recent_mentions = []
        for idx, row in fb_latest.iterrows():
            recent_mentions.append({
                "user_name": row.get("username", row.get("user_name", "Utilisateur Anonyme")),
                "comment": row.get("raw_text", row.get("feedback_text", "")),
                "sentiment": row.get("sentiment", "neutral"),
                "timestamp": row.get("date_str")
            })

        # 7. Package Unified Structural Payload
        return {
            "overall_sentiment": {
                "score_out_of_100": overall_score,
                "vs_last_week": f"+{variance_vs_last_week}" if variance_vs_last_week >= 0 else str(variance_vs_last_week)
            },
            "distribution": {
                "positive": pos,
                "neutral": neu,
                "negative": neg
            },
            "sentiment_over_time": sentiment_over_time,
            "sentiment_mix_percentage": {
                "selected_period": "All Available Records",
                "positive": f"{pos_pct}%",
                "neutral": f"{neu_pct}%",
                "negative": f"{neg_pct}%"
            },
            "recent_mentions": {
                "platform_source": "Live feed from Facebook",
                "feed": recent_mentions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate comprehensive reputation engine metrics: {str(e)}")

@app.get("/churn")
def get_churn_data_or_customer_profile(customer_id: Optional[str] = None, threshold: float = 0.40):
    """
    Dual-purpose analytical endpoint optimized for telecom scale:
    1. No parameter: Queries database, computes batch churn risk, and returns the top 10 highest risk customer profiles.
    2. With customer_id: Direct key lookup via database indexing, returning granular individual predictive metrics.
    """
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Machine learning engine weights not compiled.")
        
    if db.client is None:
        raise HTTPException(status_code=500, detail="Supabase connection target missing.")
        
    try:
        # ----------------------------------------------------
        # CASE 1: TARGETED CUSTOMER SPECIFIC QUERY (Optimized server-side filter)
        # ----------------------------------------------------
        if customer_id:
            target_id = str(customer_id).strip()
            
            # Efficient indexing lookup directly on the database instead of loading all rows
            res = db.client.table("subscribers").select("*").eq("customer_id", target_id).execute()
            
            if not res.data:
                raise HTTPException(status_code=404, detail=f"Customer ID '{target_id}' not found in database records.")
                
            matched_user = pd.DataFrame(res.data)
            
            # Process and predict exclusively for this user profile
            cleaned_df = execute_cleaning_and_quality_logs(matched_user)
            X_processed, _ = engineer_features(cleaned_df)
            
            global FEATURE_COLUMNS
            if FEATURE_COLUMNS is not None:
                X_processed = X_processed.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                
            prob = float(CHAMPION_MODEL.predict_proba(X_processed)[0][1])
            row_data = matched_user.iloc[0]
            
            return {
                "search_mode": "individual_profile_lookup",
                "customer_info": {
                    "customer_id": row_data.get("customer_id"),
                    "state_region": row_data.get("state", "unknown"),
                    "tenure_months": int(row_data.get("account_length", 0)),
                    "area_code": row_data.get("area_code"),
                    "customer_service_calls": int(row_data.get("customer_service_calls", 0)),
                    "feedback_text": row_data.get("feedback_text", ""),
                    "top_factor_category": row_data.get("feedback_category", "unknown"),
                    "sentiment_flag": row_data.get("sentiment", "unknown"),
                    "complaint_intensity_score": int(row_data.get("complaint_intensity", 0))
                },
                "predictive_analytics": {
                    "churn_probability": round(prob, 4),
                    "churn_probability_percentage": f"{round(prob * 100, 2)}%",
                    "risk_level": "High" if prob >= threshold else "Low",
                    "action_required": prob >= threshold
                }
            }

        # ----------------------------------------------------
        # CASE 2: AGGREGATED DASHBOARD ENGINE MODE (Fallback if no search ID)
        # ----------------------------------------------------
        # Note: For strict production safety at scale, consider adding a .limit(1000) here
        res = db.client.table("subscribers").select("*").execute()
        if not res.data:
            raise HTTPException(status_code=444, detail="The production database tables are currently empty.")
            
        raw_df = pd.DataFrame(res.data)
        total_customers = len(raw_df)
        
        cleaned_df = execute_cleaning_and_quality_logs(raw_df)
        X_processed, _ = engineer_features(cleaned_df)
        
        if FEATURE_COLUMNS is not None:
            X_processed = X_processed.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            
        probabilities = CHAMPION_MODEL.predict_proba(X_processed)[:, 1]
        raw_df['churn_probability'] = [round(float(p), 4) for p in probabilities]
        raw_df['risk_level'] = raw_df['churn_probability'].apply(lambda p: "High" if p >= threshold else "Low")
        
        avg_churn_risk = round(float(probabilities.mean() * 100), 2)
        high_risk_count = int((probabilities >= threshold).sum())
        
        # Sort and extract exactly the top 10 highest risk customer matrix rows
        sample_df = raw_df.sort_values(by="churn_probability", ascending=False).head(10)
        top_risk_profiles = []
        for _, row in sample_df.iterrows():
            top_risk_profiles.append({
                "customer_id": row.get("customer_id"),
                "churn_probability": row.get("churn_probability"),
                "risk_level": row.get("risk_level"),
                "top_factor_category": row.get("feedback_category", "unknown"),
                "state_region": row.get("state", "unknown"),
                "tenure_months": int(row.get("account_length", 0)),
                "customer_service_calls": int(row.get("customer_service_calls", 0)),
                "sentiment_flag": row.get("sentiment", "unknown")
            })
            
        return {
            "search_mode": "global_dashboard_summary",
            "total_customers": total_customers,
            "avg_churn_risk_percentage": f"{avg_churn_risk}%",
            "high_risk_customers": high_risk_count,
            "top_risk_profiles": top_risk_profiles
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile churn data stream: {str(e)}")



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)