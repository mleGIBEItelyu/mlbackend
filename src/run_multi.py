"""Run ensemble training + backtest for multiple tickers."""
from modeling import EnsembleModeling
from features import prepare_features
from backtest import run_financial_backtest

TICKERS = ['BBRI', 'TLKM', 'ASII']
TRAIN_CUTOFF = '2024-01-01'

for ticker in TICKERS:
    print(f"\n{'#'*60}")
    print(f"  PROCESSING: {ticker}")
    print(f"{'#'*60}")
    
    # 1. Train
    df = prepare_features(ticker, save_csv=False)
    if df is None or df.empty:
        print(f"[SKIP] No data for {ticker}")
        continue

    m = EnsembleModeling(ticker=ticker)
    X_tech, X_fund, y, df_p = m.preprocess(df)
    
    train_mask = df_p['date'] < TRAIN_CUTOFF
    test_mask = df_p['date'] >= TRAIN_CUTOFF
    
    rows_train = train_mask.sum()
    rows_test = test_mask.sum()
    
    print(f"\n  Training : {df_p.loc[train_mask, 'date'].min().date()} to {df_p.loc[train_mask, 'date'].max().date()} ({rows_train} rows)")
    print(f"  Testing  : {df_p.loc[test_mask, 'date'].min().date()} to {df_p.loc[test_mask, 'date'].max().date()} ({rows_test} rows)\n")
    
    m.train(X_tech.loc[train_mask], X_fund.loc[train_mask], y.loc[train_mask])
    m.evaluate(X_tech.loc[test_mask], X_fund.loc[test_mask], y.loc[test_mask])
    
    # 2. Backtest
    run_financial_backtest(ticker)
    print()
