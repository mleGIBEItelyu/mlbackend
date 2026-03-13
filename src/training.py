"""Run ensemble training + backtest for multiple tickers."""
import argparse
import json
import os
from modeling import EnsembleModeling
from features import prepare_features
from backtest import run_financial_backtest
from uploadHF import upload_model

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TICKERS_JSON = os.path.join(PROJECT_ROOT, 'tickers.json')

def main():
    parser = argparse.ArgumentParser(description='Run multi-ticker training and backtest')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated list of tickers. If None, uses default list.')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials for training.')
    parser.add_argument('--cutoff', type=str, default='2024-01-01',
                        help='Date cutoff for train/test split.')
    parser.add_argument('--no-backtest', action='store_true',
                        help='If set, skips the financial backtest step.')
    parser.add_argument('--upload', action='store_true',
                        help='If set, uploads trained models to Hugging Face.')
    
    args = parser.parse_args()

    # Load tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        # Default fallback
        if os.path.exists(TICKERS_JSON):
            with open(TICKERS_JSON, 'r') as f:
                tickers = json.load(f)['tickers']
        else:
            tickers = ['BBRI', 'TLKM', 'ASII']

    for ticker in tickers:
        print(f"\n{'#'*60}")
        print(f"  PROCESSING: {ticker} (Trials: {args.trials})")
        print(f"{'#'*60}")
        
        # 1. Train
        df = prepare_features(ticker, save_csv=False)
        if df is None or df.empty:
            print(f"[SKIP] No data for {ticker}")
            continue

        m = EnsembleModeling(ticker=ticker)
        X_tech, X_fund, y, df_p = m.preprocess(df)
        
        train_mask = df_p['date'] < args.cutoff
        test_mask = df_p['date'] >= args.cutoff
        
        rows_train = train_mask.sum()
        rows_test = test_mask.sum()
        
        if rows_train < 10:
            print(f"[SKIP] Not enough training data for {ticker} ({rows_train} rows)")
            continue

        print(f"\n  Training : {df_p.loc[train_mask, 'date'].min().date()} to {df_p.loc[train_mask, 'date'].max().date()} ({rows_train} rows)")
        print(f"  Testing  : {df_p.loc[test_mask, 'date'].min().date()} to {df_p.loc[test_mask, 'date'].max().date()} ({rows_test} rows)\n")
        
        m.train(X_tech.loc[train_mask], X_fund.loc[train_mask], y.loc[train_mask], n_trials=args.trials)
        m.evaluate(X_tech.loc[test_mask], X_fund.loc[test_mask], y.loc[test_mask])
        
        # 2. Upload to Hugging Face
        if args.upload:
            model_file = os.path.join(m.models_dir, f"{ticker}.pkl")
            upload_model(model_file, ticker)

        # 3. Backtest
        if not args.no_backtest:
            run_financial_backtest(ticker)
        
        print()

if __name__ == "__main__":
    main()
