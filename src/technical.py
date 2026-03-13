import os
import sys
import json
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
from dbconfig import get_supabase_client, upload_df
import indicators as ind

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TICKERS_PATH = os.path.join(PROJECT_ROOT, 'tickers.json')

def scrape_technical(ticker_list=None, period='max'):
    """
    Scrape data teknikal dan indikator.
    
    Args:
        ticker_list: List of ticker codes (e.g. ['BBCA','BBRI']).
                     Jika None, ambil semua dari tickers.json.
        period: Periode data yfinance (e.g. 'max', '7d', '1mo').
    """
    with open(TICKERS_PATH, 'r') as f:
        ticker_config = json.load(f)

    suffix = ticker_config['suffix']

    # Gunakan sub-list jika diberikan, otherwise semua ticker
    if ticker_list:
        tickers = ticker_list
    else:
        tickers = ticker_config['tickers']

    tickers_yf = [t + suffix for t in tickers]

    print(f'Starting technical scraping for {len(tickers_yf)} tickers (Period: {period})...')
    supabase = get_supabase_client()
    total_rows = 0

    for ticker_yf in tickers_yf:
        ticker_code = ticker_yf.replace(suffix, '')
        
        # Download data berdasarkan period yang diminta
        data = yf.download(ticker_yf, period=period, progress=False)
        
        if data.empty:
            print(f'[SKIP] {ticker_yf} - no data')
            continue

        # Flatten MultiIndex if necessary
        if isinstance(data.columns, pd.MultiIndex):
            if ticker_yf in data.columns.get_level_values(1):
                data = data.xs(ticker_yf, axis=1, level=1)
            else:
                data.columns = data.columns.get_level_values(0)
        
        data = data.loc[:, ~data.columns.duplicated()]

        # Calculate Indicators
        data['SMA_5'] = data['Close'].rolling(5).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI_14'] = ind.compute_rsi(data['Close'], 14)
        data['BB_upper'], data['BB_lower'] = ind.bollinger_bands(data['Close'], 20, 2)
        data['MACD'], data['MACD_signal'] = ind.macd(data['Close'])
        data['ATR_14'] = ind.compute_atr(data['High'], data['Low'], data['Close'], 14)
        data['OBV'] = ind.compute_obv(data['Close'], data['Volume'])
        data['Stoch_K'], data['Stoch_D'] = ind.compute_stochastic(data['High'], data['Low'], data['Close'])
        data['return_1d'] = data['Close'].pct_change(1)
        data['return_3d'] = data['Close'].pct_change(3)
        data['return_5d'] = data['Close'].pct_change(5)

        data.dropna(inplace=True)
        df = data.reset_index()
        
        # 1. Map Raw Technical Data (OHLCV)
        tech_map = {
            'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }
        df_tech = df[list(tech_map.keys())].rename(columns=tech_map)
        df_tech['ticker'] = ticker_code

        # 2. Map Indicator Data
        ind_map = {
            'Date': 'date',
            'SMA_5': 'sma_5', 'SMA_20': 'sma_20', 'RSI_14': 'rsi_14',
            'BB_upper': 'bb_upper', 'BB_lower': 'bb_lower',
            'MACD': 'macd', 'MACD_signal': 'macd_signal',
            'ATR_14': 'atr_14', 'OBV': 'obv',
            'Stoch_K': 'stoch_k', 'Stoch_D': 'stoch_d',
            'return_1d': 'return_1d', 'return_3d': 'return_3d', 'return_5d': 'return_5d',
        }
        df_ind = df[list(ind_map.keys())].rename(columns=ind_map)
        df_ind['ticker'] = ticker_code

        # Upload to both tables
        rows_tech = upload_df(supabase, "technical_data", df_tech, date_col='date')
        rows_ind = upload_df(supabase, "indicator_data", df_ind, date_col='date')
        
        total_rows += rows_tech
        print(f'[OK] {ticker_code}: {rows_tech} raw rows, {rows_ind} indicator rows updated')

    print(f'\nTotal technical data uploaded: {total_rows} rows')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape technical data')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated list of tickers, e.g. "BBCA,BBRI,TLKM"')
    parser.add_argument('--period', type=str, default='max',
                        help='yfinance period (e.g. "max", "7d", "1mo"). Default is "max".')
    args = parser.parse_args()

    t_list = None
    if args.tickers:
        t_list = [t.strip() for t in args.tickers.split(',')]
    
    scrape_technical(ticker_list=t_list, period=args.period)
