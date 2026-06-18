# C:\Users\chaim\Desktop\clients_prediction\processing.py
import pandas as pd
import numpy as np
from typing import Tuple, List

def execute_cleaning_and_quality_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data telemetry, logs asset shapes, and audits unparsed text properties.
    DOES NOT drop columns required by subsequent steps.
    """
    df = df.copy()
    print("\n" + "="*50 + "\n[REAL DATA DATA HEALTH AUDIT REPORT]\n" + "="*50)
    print(f"Initial Row Count Ingested: {df.shape[0]}")
    print(f"Initial Column Count Ingested: {df.shape[1]}")
    
    # Audit tracking for new LLM fields
    for col in ['feedback_category', 'sentiment', 'complaint_intensity']:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  [WARNING] Feature '{col}' contains {missing} unparsed rows.")
                
    print(f"Total Raw Null Matrix Positions: {df.isnull().sum().sum()}")
    print("="*50 + "\n")
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Isolates targets, drops non-numeric structural text arrays, and 
    factorizes all non-numeric categories into numeric indices.
    """
    df = df.copy()
    print(f"--> Initializing Feature Engineering Gate for shape: {df.shape}")

    # -------------------------------------------------------------------------
    # 1. GROUND-TRUTH TARGET ISOLATION LAYER
    # -------------------------------------------------------------------------
    if 'churn' in df.columns:
        target = df['churn'].astype(int)
        df = df.drop(columns=['churn'])
        print("    Successfully isolated target variable from 'churn'.")
    elif 'churn_flag' in df.columns:
        target = df['churn_flag'].astype(int)
        df = df.drop(columns=['churn_flag'])
        print("    Successfully isolated target variable from 'churn_flag'.")
    else:
        raise KeyError("Fatal: Extraction failed. Target column ('churn' or 'churn_flag') absent.")

    # -------------------------------------------------------------------------
    # 2. METADATA & TEXT REMOVAL LAYER
    # -------------------------------------------------------------------------
    # Drop structural tracking values, row IDs, and long raw text blocks
    cols_to_drop = ['_id', 'customer_id', 'split', 'feedback_text']
    dropped_existing = [col for col in cols_to_drop if col in df.columns]
    
    if dropped_existing:
        df = df.drop(columns=dropped_existing)
        print(f"    Dropped non-numeric tracking/identity sequences: {dropped_existing}")

    # -------------------------------------------------------------------------
    # 3. INDUSTRIAL-GRADE CATEGORICAL ENCODING LAYER
    # -------------------------------------------------------------------------
    # This catches EVERYTHING that isn't a number (strings, objects, categories)
    # and safely factorizes it into clean matrix indices for your models.
    encoded_features: List[str] = []
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna('Missing')
            # Convert text labels to discrete mathematical index positions
            df[col] = df[col].astype(str).astype('category').cat.codes
            encoded_features.append(col)
            
    if encoded_features:
        print(f"    Factorized categorical elements to numeric indices: {encoded_features}")

    # -------------------------------------------------------------------------
    # 4. DATA SANITIZATION LAYER
    # -------------------------------------------------------------------------
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        df = df.fillna(0)
        print(f"    Filled {nan_count} residual continuous gaps with default (0).")

    print(f"--> Preprocessing final matrix matrix shape: {df.shape}\n")
    return df, target