# data_loader.py
import pandas as pd
import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv() 


conn = psycopg2.connect(
    host="localhost",     
    database=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    port=5432
)



def load_real_telecom_data(file_path: str) -> pd.DataFrame:
    """
    Loads a local subscriber dataset (.csv, .xlsx, or .xls) from disk.
    """
    if not os.path.exists(file_path):
        print(f"\n[CRITICAL ERROR] File not found at target path: '{file_path}'")
        print("Please check the file name and try again.")
        sys.exit(1)
        
    print(f"Reading data from: {file_path}")
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        print("\n[CRITICAL ERROR] Unsupported file format! Please use a .csv or .xlsx file.", file=sys.stderr)
        sys.exit(1)