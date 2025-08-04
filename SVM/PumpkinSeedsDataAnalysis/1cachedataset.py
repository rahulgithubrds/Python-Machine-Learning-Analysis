# This program connects and extract Pumpkin Seeds Dataset and cache dataset into a ZIP/.pkl file and stores it

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import shutil
import joblib

# Cache directory
CACHE_DIR = "data_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "pumpkin_seeds.pkl")

def pickDataToDF(use_cache=True):
    """Fetch data with caching support"""
    # Use cached data if available
    if use_cache and os.path.exists(CACHE_FILE):
        print("Loading data from cache...")
        return joblib.load(CACHE_FILE)
    
    # --- DOWNLOAD LOGIC (OUTSIDE CACHE CHECK) ---
    api = KaggleApi()
    try:
        api.authenticate()
        print("Kaggle API authenticated successfully")

        # Dataset information
        dataset_ref = "muratkokludataset/pumpkin-seeds-dataset"
        download_dir = "temp_data"

        # Create temp directory
        os.makedirs(download_dir, exist_ok=True)
        print(f"Created temporary directory: {download_dir}")
    
        # Download dataset
        print("Downloading dataset...")
        api.dataset_download_files(
            dataset=dataset_ref,
            path=download_dir,
            unzip=False,
            force=True
        )

        # Find the downloaded zip file
        zip_files = [f for f in os.listdir(download_dir) if f.endswith('.zip')]
        if not zip_files:
            raise FileNotFoundError("No zip file found in download directory")
            
        zip_path = os.path.join(download_dir, zip_files[0])
        print(f"Found zip file: {zip_path}")
        
        # Extract zip contents
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find Excel file in archive
            excel_files = [f for f in z.namelist() if f.lower().endswith(('.xlsx', '.xls'))]
            if not excel_files:
                raise FileNotFoundError("No Excel file found in the zip archive")
                
            # Extract first Excel file found
            excel_file = excel_files[0]
            z.extract(excel_file, download_dir)
            extracted_path = os.path.join(download_dir, excel_file)
            print(f"Extracted Excel file: {extracted_path}")

        # Read Excel file
        print("Reading Excel file...")
        df = pd.read_excel(extracted_path, engine='openpyxl')
    
        # Cache the DataFrame
        os.makedirs(CACHE_DIR, exist_ok=True)
        joblib.dump(df, CACHE_FILE)
        print(f"Data cached at {CACHE_FILE}")
        
        # return df

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:
        # Cleanup temp files (keep cache)
        if 'download_dir' in locals() and os.path.exists(download_dir):
            shutil.rmtree(download_dir)

pickDataToDF()
