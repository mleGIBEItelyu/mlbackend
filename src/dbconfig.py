import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_df(supabase: Client, table_name: str, df: pd.DataFrame,
              date_col: str = 'date', batch_size: int = 500, on_conflict: str = 'date,ticker'):
    """
    Upload DataFrame ke satu tabel Supabase.
    - date_col : nama kolom tanggal yang akan dikonversi ke string ISO
    - on_conflict : kolom untuk resolusi konflik upsert
    """
    df_upload = df.copy()

    # Konversi date ke string ISO
    if date_col in df_upload.columns:
        df_upload[date_col] = pd.to_datetime(df_upload[date_col]).dt.strftime('%Y-%m-%d')

    # NaN → None biar JSON tidak error
    df_upload = df_upload.where(pd.notnull(df_upload), None)

    records = df_upload.to_dict(orient='records')

    # Sanitize: Infinity/-Infinity → None (not JSON compliant)
    import math
    for rec in records:
        for key, val in rec.items():
            if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
                rec[key] = None
    total = len(records)
    success = 0

    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        try:
            # Gunakan on_conflict agar upsert bekerja pada kombinasi kolom unik
            if table_name in ["stocks"]: # stocks table uses 'ticker' as PK
                 supabase.table(table_name).upsert(batch, on_conflict="ticker").execute()
            else:
                 supabase.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
            success += len(batch)
        except Exception as e:
            print(f'  [ERROR] {table_name} batch {i}-{i + len(batch)}: {e}')

    status = 'OK' if success == total else 'PARTIAL'
    print(f'  [{status}] {table_name}: {success}/{total} rows')
    return success
