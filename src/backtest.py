import os
import pandas as pd
import numpy as np
import joblib
from features import prepare_features

TRANSACTION_FEE = 0.0015  # 0.15% per trade (buy or sell)
FUNDAMENTAL_COLS = [
    'total_assets', 'total_liabilities', 'shareholders_equity',
    'total_liabilities_equity', 'net_cash_debt', 'cash_growth'
]

def run_financial_backtest(ticker: str, initial_capital: float = 10_000_000):
    # 1. Load Ensemble Model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(base_dir), 'models', f'{ticker}.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model for {ticker} not found. Please run modeling.py first.")
        return

    package = joblib.load(model_path)
    
    # Support both old single-model and new ensemble format
    is_ensemble = 'tech_models' in package
    
    if is_ensemble:
        tech_models = package['tech_models']
        fund_models = package['fund_models']
        tech_features = package['tech_features']
        fund_features = package['fund_features']
        tech_weight = package['tech_weight']
        fund_weight = package['fund_weight']
        n_ensemble = package['n_ensemble']
    else:
        model = package['model']
        features = package['features']

    # 2. Get Data (in-memory)
    df = prepare_features(ticker, save_csv=False)
    if df is None: return

    # 3. Preprocess
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Replicate lag & extra features (same as training)
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df['close'].shift(lag)
    
    df['volatility_5'] = df['close'].rolling(5).std()
    df['volatility_20'] = df['close'].rolling(20).std()
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # Filter backtest period: 2024-01-01 to 2025-12-31
    df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2025-12-31')]
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < 2:
        print(f"Not enough data for backtesting {ticker}.")
        return

    # 4. Generate Predictions
    if is_ensemble:
        # Pisahkan fitur teknikal dan fundamental
        fund_cols_present = [c for c in fund_features if c in df.columns]
        tech_cols = [c for c in tech_features if c in df.columns]
        
        X_tech = df[tech_cols]
        X_fund = df[fund_cols_present] if fund_cols_present else pd.DataFrame(index=df.index)
        
        # Ensemble predict: rata-rata semua model
        tech_preds = np.array([m.predict(X_tech) for m in tech_models]).mean(axis=0)
        
        if fund_models and not X_fund.empty:
            fund_preds = np.array([m.predict(X_fund) for m in fund_models]).mean(axis=0)
            preds = tech_weight * tech_preds + fund_weight * fund_preds
        else:
            preds = tech_preds
    else:
        X = df[features].copy()
        preds = model.predict(X)

    df['pred_close_t1'] = preds

    # Signal: jika prediksi harga besok > harga hari ini → beli/tahan
    df['signal'] = (df['pred_close_t1'] > df['close']).astype(int)
    
    # 5. REALISTIC Financial Simulation
    balance = initial_capital
    shares = 0.0
    position = 0
    total_trades = 0
    total_fees = 0.0
    history = []

    model_type = f"Ensemble ({n_ensemble}x2)" if is_ensemble else "Single Model"

    print(f"\n{'='*50}")
    print(f"  BACKTEST SIMULATION: {ticker}")
    print(f"{'='*50}")
    print(f"Model Type      : {model_type}")
    if is_ensemble:
        print(f"Tech Weight     : {tech_weight*100:.0f}%")
        print(f"Fund Weight     : {fund_weight*100:.0f}%")
    print(f"Initial Capital : Rp{initial_capital:,.0f}")
    print(f"Period          : {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(f"Transaction Fee : {TRANSACTION_FEE*100:.2f}% per trade")
    print(f"Data Points     : {len(df)} trading days")
    print(f"{'='*50}")

    for i in range(len(df) - 1):
        today_close = df.loc[i, 'close']
        signal = df.loc[i, 'signal']
        next_open = df.loc[i + 1, 'open']

        if signal == 1 and position == 0:
            fee = balance * TRANSACTION_FEE
            total_fees += fee
            invest_amount = balance - fee
            shares = invest_amount / next_open
            balance = 0
            position = 1
            total_trades += 1
            
        elif signal == 0 and position == 1:
            gross = shares * next_open
            fee = gross * TRANSACTION_FEE
            total_fees += fee
            balance = gross - fee
            shares = 0
            position = 0
            total_trades += 1
        
        port_value = balance + (shares * today_close)
        history.append({'date': df.loc[i, 'date'], 'value': port_value})

    # Final portfolio value
    last_close = df.iloc[-1]['close']
    final_value = balance + (shares * last_close)
    
    # Buy & Hold comparison
    first_open = df.iloc[0]['open']
    bh_fee = initial_capital * TRANSACTION_FEE
    bh_shares = (initial_capital - bh_fee) / first_open
    bh_final = bh_shares * last_close

    strategy_return = ((final_value / initial_capital) - 1) * 100
    bh_return = ((bh_final / initial_capital) - 1) * 100

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"Strategy Final  : Rp{final_value:,.0f} ({strategy_return:+.2f}%)")
    print(f"Buy & Hold Final: Rp{bh_final:,.0f} ({bh_return:+.2f}%)")
    print(f"Total Trades    : {total_trades}")
    print(f"Total Fees Paid : Rp{total_fees:,.0f}")
    print(f"{'='*50}")
    
    if strategy_return > bh_return:
        print(f"[WIN] Strategy BEATS Buy & Hold by {strategy_return - bh_return:.2f}%")
    else:
        print(f"[LOSS] Strategy LOSES to Buy & Hold by {bh_return - strategy_return:.2f}%")

    return final_value

if __name__ == "__main__":
    run_financial_backtest("BBCA")
