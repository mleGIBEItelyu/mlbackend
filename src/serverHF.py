import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from features import prepare_features
from uploadHF import download_model

# ============================================================
#  KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TICKERS_PATH = os.path.join(PROJECT_ROOT, 'tickers.json')

def generate_daily_signals():
    """Membaca semua model yang tersedia dan memberikan signal harian."""
    
    # 1. Load semua ticker
    try:
        with open(TICKERS_PATH, 'r') as f:
            tickers = json.load(f)['tickers']
    except Exception as e:
        print(f"[ERROR] Gagal membaca tickers.json: {e}")
        return

    results = []
    
    print(f"\n{'='*75}")
    print(f"  STOCK FORECASTING DASHBOARD (7-DAY RANGE) - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*75}")
    print(f"{'Ticker':<8} | {'Last Close':<10} | {'7D Return':<10} | {'Range (%)':<15} | {'Signal':<8} | {'Conf':<6}")
    print(f"{'-'*85}")

    for ticker in tickers:
        model_path = os.path.join(MODELS_DIR, f"{ticker}.pkl")
        
        # Cek apakah model untuk ticker ini sudah dilatih (jika tidak, coba download dari HF)
        if not os.path.exists(model_path):
            downloaded = download_model(ticker, MODELS_DIR)
            if not downloaded:
                continue
            model_path = downloaded
            
        try:
            # 2. Load Ensemble Model
            package = joblib.load(model_path)
            tech_models = package['tech_models']
            fund_models = package['fund_models']
            tech_features = package['tech_features']
            fund_features = package['fund_features']
            tech_weight = package['tech_weight']
            fund_weight = package['fund_weight']
            
            # 3. Ambil data terbaru (termasuk hari ini)
            df = prepare_features(ticker, save_csv=False)
            if df is None or df.empty:
                continue
                
            # 4. Preprocess (persis seperti modeling/backtest)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Tambahkan fitur lag & teknikal tambahan
            for lag in [1, 2, 3, 5, 10]:
                df[f"lag_{lag}"] = df['close'].shift(lag)
            
            df['volatility_5'] = df['close'].rolling(5).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            
            # Ambil baris terakhir (Hari ini)
            latest_data = df.iloc[[-1]].copy()
            
            # 5. Prediksi (Hasil adalah % Return 7-Hari dalam desimal)
            X_tech = latest_data[tech_features]
            
            # Fallback jika fundamental tidak ada
            fund_cols_present = [c for c in fund_features if c in latest_data.columns]
            X_fund = latest_data[fund_cols_present] if fund_cols_present else pd.DataFrame()
            
            # Prediksi Ensemble (rata-rata % Return)
            all_tech_preds = np.array([m.predict(X_tech)[0] for m in tech_models])
            tech_pred_avg = all_tech_preds.mean()
            
            if fund_models and not X_fund.empty:
                all_fund_preds = np.array([m.predict(X_fund)[0] for m in fund_models])
                fund_pred_avg = all_fund_preds.mean()
                final_return_pred = tech_weight * tech_pred_avg + fund_weight * fund_pred_avg
                
                # Agreement: berapa % model setuju arah harganya (Return > 0)
                tech_up = (all_tech_preds > 0).mean()
                fund_up = (all_fund_preds > 0).mean()
                agreement = (tech_weight * tech_up + fund_weight * fund_up) * 100
            else:
                final_return_pred = tech_pred_avg
                agreement = (all_tech_preds > 0).mean() * 100
            
            # 6. Range Calculation (+/- MAE)
            mae_val = package.get('mae_score', 0.02) # Default 2% jika model lama
            range_low_val = final_return_pred - mae_val
            range_high_val = final_return_pred + mae_val
            
            range_low_pct = range_low_val * 100
            range_high_pct = range_high_val * 100
            pred_close_7d = latest_data['close'].values[0] * (1 + final_return_pred)
            
            # 7. Conservative Signal Logic
            # BUY if even the worst-case (low) is positive
            if range_low_val > 0:
                signal = "BUY"
            # SELL if even the best-case (high) is negative
            elif range_high_val < 0:
                signal = "SELL"
            # CROSS ZERO: Wait and see (Uncertainty area)
            else:
                signal = "WAIT&SEE"
                
            # 8. Formatting & Display
            range_str = f"{range_low_pct:>+5.1f}% to {range_high_pct:>+5.1f}%"
            last_close = latest_data['close'].values[0]
            date_str = latest_data['date'].iloc[0].strftime('%Y-%m-%d')
            
            print(f"{ticker:<8} | {last_close:>10,.0f} | {pred_close_7d:>10,.0f} | {range_str:<15} | {signal:<8} | {agreement:>5.0f}%")
            
            results.append({
                'ticker': ticker,
                'date': date_str,
                'last_close': float(last_close),
                'pred_7d_close': float(pred_close_7d),
                'return_7d_pct': float(final_return_pred * 100),
                'range_min_pct': float(range_low_pct),
                'range_max_pct': float(range_high_pct),
                'signal': signal,
                'confidence': float(agreement)
            })
            
        except Exception as e:
            # print(f"[WARN] Error pada {ticker}: {e}")
            continue

    print(f"{'='*75}")
    
    # Simpan hasil ke JSON dashboard jika perlu
    if results:
        output_path = os.path.join(PROJECT_ROOT, 'data', 'daily_signals.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Signals saved to: {output_path}")

if __name__ == "__main__":
    generate_daily_signals()
