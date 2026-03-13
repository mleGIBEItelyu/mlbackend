import os
import json
import time
import requests
import pandas as pd
from dbconfig import get_supabase_client, upload_df

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TICKERS_PATH = os.path.join(PROJECT_ROOT, 'tickers.json')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Referer': 'https://stockanalysis.com/',
}

# Metrik yang ingin diambil dari Balance Sheet (key di JSON → nama kolom DB)
METRICS_MAP = {
    'assets':             'total_assets',
    'liabilitiesBank':    'total_liabilities',
    'equity':             'shareholders_equity',
    'liabilitiesequity':  'total_liabilities_equity',
    'netcash':            'net_cash_debt',
    'netCashGrowth':      'cash_growth',
}

def parse_sveltekit_data(json_data: dict, metrics_map: dict) -> pd.DataFrame:
    """
    Parse SvelteKit __data.json response untuk mengekstrak data kuartalan.
    
    Struktur JSON:
    - nodes[2].data berisi info keuangan
    - nodes[2].data[7] berisi object dengan key = metric_id, value = index ke array data
    """
    try:
        fin_node = json_data['nodes'][2]['data']
    except (KeyError, IndexError):
        print("  [ERROR] Unexpected JSON structure.")
        return pd.DataFrame()

    data_array = fin_node  # Flat array yang berisi semua data

    # 1. Ambil mapping field: { metric_key: index_ke_array_tanggal/nilai }
    field_map = data_array[7]  # Object yang berisi mapping
    if not isinstance(field_map, dict):
        print("  [ERROR] Field map not found at expected position.")
        return pd.DataFrame()

    # 2. Ambil array tanggal (datekey)
    datekey_index = field_map.get('datekey')
    if datekey_index is None:
        print("  [ERROR] 'datekey' not found in field map.")
        return pd.DataFrame()

    date_indices = data_array[datekey_index]
    dates = [data_array[i] for i in date_indices]

    # 3. Ambil setiap metrik (dengan fallback untuk bank vs non-bank)
    result = {'date': dates}

    for json_key, db_col in metrics_map.items():
        # Coba key utama, lalu fallback
        actual_key = json_key
        if json_key not in field_map:
            # Fallback: bank pakai 'liabilitiesBank', non-bank pakai 'liabilities'
            fallbacks = {
                'liabilitiesBank': 'liabilities',
                'liabilities': 'liabilitiesBank',
            }
            fallback = fallbacks.get(json_key)
            if fallback and fallback in field_map:
                actual_key = fallback
            else:
                result[db_col] = [None] * len(dates)
                continue

        metric_index = field_map[actual_key]
        value_indices = data_array[metric_index]

        if isinstance(value_indices, list) and len(value_indices) == len(dates):
            values = []
            for vi in value_indices:
                val = data_array[vi]
                # Handle Infinity, -Infinity, NaN yang tidak JSON-compliant
                if val is not None and isinstance(val, float):
                    import math
                    if math.isinf(val) or math.isnan(val):
                        val = None
                values.append(val)
            result[db_col] = values
        else:
            result[db_col] = [None] * len(dates)

    df = pd.DataFrame(result)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    return df


def fetch_balance_sheet(ticker: str) -> pd.DataFrame:
    """
    Mengambil data Balance Sheet kuartalan untuk satu ticker
    menggunakan API internal StockAnalysis.com.
    """
    url = f"https://stockanalysis.com/quote/idx/{ticker}/financials/balance-sheet/__data.json?p=quarterly"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        json_data = resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"  [ERROR] HTTP {resp.status_code} for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  [ERROR] Request failed for {ticker}: {e}")
        return pd.DataFrame()

    df = parse_sveltekit_data(json_data, METRICS_MAP)

    if df.empty:
        print(f"  [WARN] No data parsed for {ticker}")
        return df

    df['ticker'] = ticker
    return df


def get_latest_date_in_db(supabase, ticker: str):
    """
    Ambil tanggal data fundamental terbaru untuk ticker tertentu dari Supabase.
    Return None jika belum ada data.
    """
    try:
        res = (supabase.table("fundamental_data")
               .select("date")
               .eq("ticker", ticker)
               .order("date", desc=True)
               .limit(1)
               .execute())
        if res.data:
            return pd.to_datetime(res.data[0]['date'])
    except Exception as e:
        print(f"  [WARN] DB check failed for {ticker}: {e}")
    return None


def scrape_fundamental(smart_mining=False):
    """
    Scrape data fundamental Balance Sheet untuk semua ticker.
    
    Args:
        smart_mining: Jika True, hanya scrape ticker yang punya data lebih baru
                      di web dibanding database. Cocok untuk cron bulanan.
    """
    with open(TICKERS_PATH, 'r') as f:
        tickers = json.load(f)['tickers']

    supabase = get_supabase_client()
    total_rows = 0
    skipped = 0
    updated = 0

    mode_label = "SMART MINING" if smart_mining else "FULL SCRAPE"
    print(f"Starting fundamental scraping for {len(tickers)} tickers...")
    print(f"Mode: {mode_label}")
    print(f"Metrics: {list(METRICS_MAP.values())}")
    print(f"Source: StockAnalysis.com (Internal API)")
    print(f"Period: Quarterly | Format: Raw\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing: {ticker}...")

        df = fetch_balance_sheet(ticker)

        if df.empty:
            print(f"  [SKIP] No data from API for {ticker}")
            skipped += 1
            time.sleep(1)
            continue

        # Smart Mining: cek apakah ada data baru
        if smart_mining:
            latest_web_date = df['date'].max()
            latest_db_date = get_latest_date_in_db(supabase, ticker)

            if latest_db_date and latest_web_date <= latest_db_date:
                print(f"  [SKIP] {ticker}: No new data (DB: {latest_db_date.date()}, Web: {latest_web_date.date()})")
                skipped += 1
                time.sleep(1)
                continue
            else:
                db_label = latest_db_date.date() if latest_db_date else "N/A"
                print(f"  [NEW]  {ticker}: New data found! (DB: {db_label}, Web: {latest_web_date.date()})")

        # Upload ke Supabase
        rows = upload_df(supabase, "fundamental_data", df, date_col='date')
        total_rows += rows
        updated += 1
        print(f"  [OK] {ticker}: {rows} rows ({df['date'].min().date()} to {df['date'].max().date()})")

        # Rate limiting: jeda 1 detik antar request agar tidak di-block
        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"  FUNDAMENTAL SCRAPING SUMMARY")
    print(f"{'='*50}")
    print(f"  Mode    : {mode_label}")
    print(f"  Updated : {updated} tickers")
    print(f"  Skipped : {skipped} tickers (no new data)")
    print(f"  Rows    : {total_rows} total rows uploaded")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape fundamental data')
    parser.add_argument('--smart', action='store_true',
                        help='Enable smart mining: only scrape if new data exists')
    args = parser.parse_args()

    scrape_fundamental(smart_mining=args.smart)

