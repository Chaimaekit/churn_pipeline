# data_loader.py
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[WARNING] Supabase environment variables missing. Supabase functions will fail.")
    supabase: Client = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_local_training_data(file_path: str = "data/train_historical.csv") -> pd.DataFrame:
    """
    Reads historical training data safely from local disk.
    This data NEVER mixes with live incoming production databases.
    """
    if not os.path.exists(file_path):
        print(f"\n[CRITICAL ERROR] Isolated Training file not found at: '{file_path}'")
        sys.exit(1)
        
    print(f"[TRAIN ISOLATION] Ingesting local historical training matrix: {file_path}")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


def load_supabase_test_data() -> pd.DataFrame:
    """
    Fetches unseen subscriber records from the Supabase production database 
    to be passed into the pipeline for validation, auditing, or live testing.
    """
    if not supabase:
        raise ValueError("Supabase client is not initialized.")

    print("[TEST ISOLATION] Fetching evaluation targets live from Supabase 'subscribers' table...")
    
    # Query all subscriber data
    response = supabase.table("subscribers").select("*").execute()
    
    if not response.data:
        print("[WARNING] Supabase 'subscribers' table is empty!")
        return pd.DataFrame()
        
    df = pd.DataFrame(response.data)
    return df