# processing.py
import pandas as pd
import numpy as np
from typing import Tuple, List

def execute_cleaning_and_quality_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data telemetry, logs asset shapes, and audits properties.
    """
    if df.empty:
        return df
        
    df = df.copy()
    print("\n" + "="*50 + "\n[DATA HEALTH AUDIT REPORT]\n" + "="*50)
    print(f"Row Count Ingested: {df.shape[0]}")
    print(f"Column Count Ingested: {df.shape[1]}")
    
    # Track any missingness in analytical columns
    for col in ['feedback_category', 'sentiment', 'complaint_intensity']:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  [WARNING] Feature '{col}' contains {missing} unparsed rows.")
                
    print(f"Total Raw Null Positions: {df.isnull().sum().sum()}")
    print("="*50 + "\n")
    return df


def engineer_features(df: pd.DataFrame, is_inference: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drops non-numeric IDs, handles missing values, and factorizes 
    categorical columns into discrete indices.
    """
    df = df.copy()
    
    # Track and split target label if it exists (training phase)
    target_col = 'churn' if 'churn' in df.columns else ('churn_flag' if 'churn_flag' in df.columns else None)
    
    y = df[target_col].copy() if (target_col and not is_inference) else None
    
    # Columns to explicitly exclude from features
    drop_cols = ['id', 'customer_id', 'created_at', 'updated_at', 'feedback_text', 'raw_text']
    if target_col:
        drop_cols.append(target_col)
        
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Industrial-Grade Categorical Encoding Layer
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna('Missing')
            df[col] = df[col].astype(str).astype('category').cat.codes
            
    # Continuous variable sanitization
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        df = df.fillna(0)
        
    return df, y