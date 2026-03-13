import os
import pandas as pd
import numpy as np
from dbconfig import get_supabase_client

def prepare_features(ticker_symbol: str, save_csv: bool = True):
    supabase = get_supabase_client()
    print(f"Preparing features for {ticker_symbol}...")
    
    # 1. Get Technical Data (Raw OHLCV)
    def fetch_all(table_name, ticker):
        all_data = []
        offset = 0
        chunk_size = 1000
        while True:
            res = supabase.table(table_name).select("*").eq("ticker", ticker).order("date", desc=True).range(offset, offset + chunk_size - 1).execute()
            if not res.data: break
            all_data.extend(res.data)
            if len(res.data) < chunk_size: break
            offset += chunk_size
        return pd.DataFrame(all_data)

    print(f"Fetching raw technical data for {ticker_symbol}...")
    df_raw = fetch_all("technical_data", ticker_symbol)
    
    print(f"Fetching indicator data for {ticker_symbol}...")
    df_ind = fetch_all("indicator_data", ticker_symbol)

    if df_raw.empty:
        print(f"[SKIP] No technical data for {ticker_symbol}")
        return None
    
    # Merge Raw and Indicators
    if not df_ind.empty:
        # Avoid duplicate columns during merge
        common_cols = ['date', 'ticker', 'created_at', 'id']
        df_ind_clean = df_ind.drop(columns=[c for c in common_cols if c in df_ind.columns and c != 'date' and c != 'ticker'], errors='ignore')
        df_tech = pd.merge(df_raw, df_ind_clean, on=['date', 'ticker'], how='left')
    else:
        df_tech = df_raw

    df_tech['date'] = pd.to_datetime(df_tech['date']).dt.tz_localize(None)
    df_tech = df_tech.sort_values('date')
    
    # 2. Get Fundamental Data
    print(f"Fetching fundamental data for {ticker_symbol}...")
    df_fund = fetch_all("fundamental_data", ticker_symbol)
    
    if not df_fund.empty:
        df_fund['date'] = pd.to_datetime(df_fund['date']).dt.tz_localize(None)
        df_fund = df_fund.sort_values('date')
        # Merge with forward fill logic
        df_combined = pd.merge_asof(df_tech, df_fund, on='date', by='ticker', direction='backward')
        df_combined = df_combined.sort_values('date').ffill().fillna(0)
    else:
        print(f"No fundamental data for {ticker_symbol}, using tech/ind only.")
        df_combined = df_tech

    # 3. Clean
    cols_to_drop = ['id', 'created_at', 'id_x', 'id_y']
    df_combined = df_combined.drop(columns=[c for c in cols_to_drop if c in df_combined.columns], errors='ignore')
    df_combined = df_combined.dropna(axis=1, how='all')
    
    # Save (optional)
    if save_csv:
        os.makedirs('data/processed', exist_ok=True)
        save_path = f'data/processed/features_{ticker_symbol}.csv'
        df_combined.to_csv(save_path, index=False)
        print(f"[OK] Saved features: {save_path} ({len(df_combined)} rows)")
    
    return df_combined

if __name__ == "__main__":
    import json
    # Use tickers.json as source
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    with open(os.path.join(PROJECT_ROOT, 'tickers.json'), 'r') as f:
        tickers = json.load(f)['tickers']
    
    # For speed in testing, we can do 1 or all
    prepare_features("BBCA")

