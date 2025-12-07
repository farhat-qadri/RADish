#    ___    _   ___  _    _    
#   | _ \  /_\ |   \(_)__| |_  
#   |   / / _ \| |) | (_-< ' \ 
#   |_|_\/_/ \_\___/|_/__/_||_| V1
#                               
#   R A D I S H | RISK ANALYSIS DASHBOARD
#   CS4001 DATA VISUALIZATION DESIGN - PROJECT 
#   TEAM OPPORTUNISTS | Shreya Farhat Puneet
#   SEP 2025 
# ----------------------------------------------
#   Support: radish@ds.study.iitm.ac.in
# ----------------------------------------------

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_PATH = "data/"
SETTINGS_FILE = os.path.join(DATA_PATH, "settings.json")
LOG_FILE = os.path.join(DATA_PATH, "cleaning_log.json")

# ---------------------------------------------------------
# PB 20251206 SETTINGS & LOGGING
# ---------------------------------------------------------
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {"last_refresh": "Never", "version": "1.0", "include_unknown": True, "currency": "$"}

def save_setting(key, value):
    settings = load_settings()
    settings[key] = value
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

def get_currency_symbol():
    """Returns the selected currency symbol from settings."""
    settings = load_settings()
    curr_map = {
        "Dollar ($)": "$", "Pound (£)": "£", "Euro (€)": "€", 
        "Rupee (₹)": "₹", "Yen (¥)": "¥", "None": ""
    }
    # Handle both full name (from UI) and symbol (stored)
    val = settings.get("currency", "$")
    return curr_map.get(val, val) if len(val) > 1 else val

def get_cleaning_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return ["No logs available. Run processing first."]

def _save_log(log_list):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_list, f, indent=4)

# ---------------------------------------------------------
# PB 20251204 DATA OPERATIONS
# ---------------------------------------------------------
def check_data_status():
    files = {
        "Raw - Application": "application_data (sample).csv",
        "Raw - Previous App": "previous_application (sample).csv",
        "Cleaned - Merged": "radish_merged_data.csv"
    }
    status = {}
    for label, filename in files.items():
        exists = os.path.exists(os.path.join(DATA_PATH, filename))
        status[label] = "✅ Available" if exists else "❌ Missing"
    return status

def load_data():
    """Loads data and applies global filters (Preferences)."""
    merged_path = os.path.join(DATA_PATH, "radish_merged_data.csv")
    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path)
        
        # APPLY PREFERENCES
        settings = load_settings()
        if not settings.get("include_unknown", True):
            # Filter out 'Unknown' from key demographics
            # We filter robustly to avoid errors if cols are missing
            if 'CODE_GENDER' in df.columns:
                df = df[df['CODE_GENDER'] != 'Unknown']
                df = df[df['CODE_GENDER'] != 'XNA']
            if 'NAME_FAMILY_STATUS' in df.columns:
                df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']
                
        return df
    else:
        return None

def get_applicant_history(sk_id):
    try:
        prev_path = os.path.join(DATA_PATH, "previous_application (sample).csv")
        if os.path.exists(prev_path):
            prev = pd.read_csv(prev_path)
            return prev[prev['SK_ID_CURR'] == sk_id]
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching history: {e}")
        return pd.DataFrame()

def process_and_merge():
    log = []
    log.append(f"⏱️ Pipeline started at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        app_path = os.path.join(DATA_PATH, "application_data (sample).csv")
        prev_path = os.path.join(DATA_PATH, "previous_application (sample).csv")
        
        if not os.path.exists(app_path) or not os.path.exists(prev_path):
            log.append("❌ Error: Source files not found.")
            _save_log(log)
            return None

        app = pd.read_csv(app_path)
        prev = pd.read_csv(prev_path)
        log.append(f"✅ Loaded Sources: App ({app.shape[0]} rows), Prev ({prev.shape[0]} rows)")

        log.append("🧹 Starting Categorical Cleanup (XNA/NaN -> 'Unknown')...")
        cat_cols = app.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            mask_xna = app[col].isin(['XNA', 'xna', ''])
            count_xna = mask_xna.sum()
            count_nan = app[col].isna().sum()
            
            if count_xna > 0 or count_nan > 0:
                app.loc[mask_xna, col] = np.nan
                app[col] = app[col].fillna('Unknown')
                log.append(f"   > Fixed '{col}': Imputed {count_xna} 'XNA's and {count_nan} NaNs")

        log.append("🔗 Aggregating History...")
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'AMT_CREDIT': 'mean',
            'NAME_CONTRACT_STATUS': lambda x: (x == 'Refused').sum()
        }).rename(columns={
            'SK_ID_PREV': 'PREV_APP_COUNT',
            'AMT_CREDIT': 'AVG_PREV_CREDIT',
            'NAME_CONTRACT_STATUS': 'PREV_REFUSALS'
        }).reset_index()
        
        log.append("🔄 Merging Datasets...")
        df = app.merge(prev_agg, on='SK_ID_CURR', how='left')
        
        log.append("🛠️ Engineering Features...")
        df['PREV_APP_COUNT'] = df['PREV_APP_COUNT'].fillna(0)
        df['PREV_REFUSALS'] = df['PREV_REFUSALS'].fillna(0)
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        
        df.to_csv(os.path.join(DATA_PATH, "radish_merged_data.csv"), index=False)
        log.append("💾 Saved 'radish_merged_data.csv'")
        
        current_time = datetime.now().strftime("%a, %d %b %Y, %H:%M:%S")
        save_setting("last_refresh", current_time)
        
        log.append("✅ COMPLETED SUCCESSFULLY")
        _save_log(log) 
        
        return df
    except Exception as e:
        log.append(f"❌ CRITICAL ERROR: {str(e)}")
        _save_log(log)
        print(f"Error: {e}")
        return None