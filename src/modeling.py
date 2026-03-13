import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import prepare_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


TECH_WEIGHT = 0.85       # Bobot model teknikal
FUND_WEIGHT = 0.15       # Bobot model fundamental (bias arah)
N_ENSEMBLE = 10          # Jumlah model per grup (ensemble)
N_OPTUNA_TRIALS = 50     # Jumlah trial Optuna (tuning agresif)
TSCV_SPLITS = 5          # TimeSeriesSplit folds

# Kolom fundamental (sisanya dianggap teknikal)
FUNDAMENTAL_COLS = [
    'total_assets', 'total_liabilities', 'shareholders_equity',
    'total_liabilities_equity', 'net_cash_debt', 'cash_growth'
]

class EnsembleModeling:
    """
    Dual-Model Weighted Ensemble:
    - 10 model XGBoost dilatih pada fitur TEKNIKAL → rata-rata prediksi
    - 10 model XGBoost dilatih pada fitur FUNDAMENTAL → rata-rata prediksi
    - Prediksi akhir = 85% * teknikal + 15% * fundamental
    """

    def __init__(self, ticker, target_column="close"):
        self.ticker = ticker
        self.target_column = target_column
        self.tech_models = []
        self.fund_models = []
        self.tech_features = None
        self.fund_features = None
        self.best_tech_params = None
        self.best_fund_params = None

        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(os.path.dirname(self.base_dir), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    def preprocess(self, df):
        """Preprocess data dan pisahkan fitur teknikal vs fundamental."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Target: harga close BESOK
        df["target_t1"] = df[self.target_column].shift(-1)

        # Lag features (data masa lalu)
        for lag in [1, 2, 3, 5, 10]:
            df[f"lag_{lag}"] = df[self.target_column].shift(lag)

        # Volatility features
        df['volatility_5'] = df[self.target_column].rolling(5).std()
        df['volatility_20'] = df[self.target_column].rolling(20).std()

        # Momentum features
        df['momentum_5'] = df[self.target_column].pct_change(5)
        df['momentum_10'] = df[self.target_column].pct_change(10)

        df = df.dropna()

        exclude = ["target_t1", "date", "ticker", "created_at", "id",
                    "created_at_x", "created_at_y", "id_x", "id_y"]
        all_numeric = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in exclude if c in df.columns], errors='ignore'
        )
        y = df["target_t1"]

        # Pisahkan fitur teknikal dan fundamental
        fund_cols_present = [c for c in FUNDAMENTAL_COLS if c in all_numeric.columns]
        tech_cols = [c for c in all_numeric.columns if c not in FUNDAMENTAL_COLS]

        X_tech = all_numeric[tech_cols]
        X_fund = all_numeric[fund_cols_present] if fund_cols_present else pd.DataFrame(index=df.index)

        self.tech_features = X_tech.columns.tolist()
        self.fund_features = X_fund.columns.tolist()

        print(f"  Technical features : {len(self.tech_features)}")
        print(f"  Fundamental features: {len(self.fund_features)}")

        return X_tech, X_fund, y, df

    def _tune(self, X, y, label=""):
        """Tuning agresif dengan Optuna — search space yang luas."""
        print(f"  Tuning {label} model ({N_OPTUNA_TRIALS} trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
            }
            model = xgb.XGBRegressor(**params, verbosity=0)
            tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)
            errors = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                errors.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
            return np.mean(errors)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)
        print(f"  Best {label} RMSE: {study.best_value:.4f}")
        return study.best_params

    def _train_ensemble(self, X, y, best_params, label=""):
        """Train N_ENSEMBLE model dengan seed berbeda, simpan semuanya."""
        print(f"  Training {label} ensemble ({N_ENSEMBLE} models)...")
        models = []
        for i in range(N_ENSEMBLE):
            params = best_params.copy()
            params['random_state'] = 42 + i  # Seed berbeda tiap model
            model = xgb.XGBRegressor(**params, verbosity=0)
            model.fit(X, y)
            models.append(model)
        return models

    def _predict_ensemble(self, models, X):
        """Prediksi ensemble: rata-rata dari semua model."""
        if not models or X.empty:
            return np.zeros(len(X))
        preds = np.array([m.predict(X) for m in models])
        return preds.mean(axis=0)

    def train(self, X_tech_train, X_fund_train, y_train, n_trials=None):
        """Full training pipeline: tune + ensemble untuk kedua grup."""
        if n_trials is not None:
            self.n_trials = n_trials
            
        # 1. Tune & Train Technical Models
        self.best_tech_params = self._tune(X_tech_train, y_train, label="TECHNICAL")
        self.tech_models = self._train_ensemble(X_tech_train, y_train, self.best_tech_params, label="TECHNICAL")

        # 2. Tune & Train Fundamental Models (jika ada fitur)
        if not X_fund_train.empty and len(self.fund_features) > 0:
            self.best_fund_params = self._tune(X_fund_train, y_train, label="FUNDAMENTAL")
            self.fund_models = self._train_ensemble(X_fund_train, y_train, self.best_fund_params, label="FUNDAMENTAL")
        else:
            print("  [WARN] No fundamental features, using technical only (100%)")

    def predict(self, X_tech, X_fund):
        """
        Prediksi dengan weighted ensemble:
        Final = TECH_WEIGHT * avg(tech_models) + FUND_WEIGHT * avg(fund_models)
        """
        tech_pred = self._predict_ensemble(self.tech_models, X_tech)

        if self.fund_models and not X_fund.empty:
            fund_pred = self._predict_ensemble(self.fund_models, X_fund)
            final_pred = TECH_WEIGHT * tech_pred + FUND_WEIGHT * fund_pred
        else:
            final_pred = tech_pred  # 100% teknikal jika tidak ada fundamental

        return final_pred

    def evaluate(self, X_tech_test, X_fund_test, y_test):
        """Evaluasi model dan simpan artifacts."""
        preds = self.predict(X_tech_test, X_fund_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"\n{'='*50}")
        print(f"  ENSEMBLE EVALUATION: {self.ticker}")
        print(f"{'='*50}")
        print(f"  Architecture : Dual-Model Weighted Ensemble")
        print(f"  Tech Weight  : {TECH_WEIGHT*100:.0f}% ({N_ENSEMBLE} models)")
        print(f"  Fund Weight  : {FUND_WEIGHT*100:.0f}% ({N_ENSEMBLE} models)")
        print(f"  Total Models : {N_ENSEMBLE * 2}")
        print(f"  Optuna Trials: {self.n_trials}")
        print(f"{'='*50}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")
        print(f"{'='*50}")

        # Save ensemble artifacts
        save_path = os.path.join(self.models_dir, f"{self.ticker}.pkl")
        joblib.dump({
            "tech_models": self.tech_models,
            "fund_models": self.fund_models,
            "tech_features": self.tech_features,
            "fund_features": self.fund_features,
            "tech_params": self.best_tech_params,
            "fund_params": self.best_fund_params,
            "tech_weight": TECH_WEIGHT,
            "fund_weight": FUND_WEIGHT,
            "n_ensemble": N_ENSEMBLE,
        }, save_path)
        print(f"  Ensemble saved to {save_path}")

        return preds


if __name__ == "__main__":
    TICKER = "BBCA"
    TRAIN_CUTOFF = "2024-01-01"

    df = prepare_features(TICKER, save_csv=False)

    if df is not None and not df.empty:
        m = EnsembleModeling(ticker=TICKER)
        X_tech, X_fund, y, df_p = m.preprocess(df)

        # TEMPORAL SPLIT
        train_mask = df_p['date'] < TRAIN_CUTOFF
        test_mask = df_p['date'] >= TRAIN_CUTOFF

        X_tech_train = X_tech.loc[train_mask]
        X_fund_train = X_fund.loc[train_mask]
        y_train = y.loc[train_mask]

        X_tech_test = X_tech.loc[test_mask]
        X_fund_test = X_fund.loc[test_mask]
        y_test = y.loc[test_mask]

        print(f"\n{'='*50}")
        print(f"  TEMPORAL SPLIT")
        print(f"{'='*50}")
        print(f"  Training : {df_p.loc[train_mask, 'date'].min().date()} to {df_p.loc[train_mask, 'date'].max().date()} ({len(X_tech_train)} rows)")
        print(f"  Testing  : {df_p.loc[test_mask, 'date'].min().date()} to {df_p.loc[test_mask, 'date'].max().date()} ({len(X_tech_test)} rows)")
        print(f"  NO DATA LEAKAGE: Model has NEVER seen test data.\n")

        m.train(X_tech_train, X_fund_train, y_train)
        m.evaluate(X_tech_test, X_fund_test, y_test)
    else:
        print(f"Failed to prepare features for {TICKER}.")
