import os
import pandas as pd

# Path when running INSIDE the Docker container
file_path = "/tmp/facebook_scraped_data.xlsx"

if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} rows from n8n successfully!")
    print(df.head())
else:
    print(f"File not found yet at {file_path}. Please execute the n8n workflow first.")