-- ============================================================
-- MLE Stock Forecasting - Complete Database Schema
-- Paste seluruh isi file ini di Supabase SQL Editor → Run
-- ============================================================

-- 1. Tabel Master: Daftar Emiten
CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Tabel Data Teknikal Mentah (OHLCV)
CREATE TABLE IF NOT EXISTS technical_data (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

-- 3. Tabel Indikator Teknikal (Hasil Perhitungan)
CREATE TABLE IF NOT EXISTS indicator_data (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    sma_5 DOUBLE PRECISION,
    sma_20 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    obv DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    return_1d DOUBLE PRECISION,
    return_3d DOUBLE PRECISION,
    return_5d DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

-- 4. Tabel Data Fundamental (Balance Sheet - Quarterly, Raw)
-- Source: StockAnalysis.com Internal API
CREATE TABLE IF NOT EXISTS fundamental_data (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    total_assets DOUBLE PRECISION,
    total_liabilities DOUBLE PRECISION,
    shareholders_equity DOUBLE PRECISION,
    total_liabilities_equity DOUBLE PRECISION,
    net_cash_debt DOUBLE PRECISION,
    cash_growth DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

-- 5. Index untuk mempercepat query
CREATE INDEX IF NOT EXISTS idx_technical_ticker_date ON technical_data(ticker, date);
CREATE INDEX IF NOT EXISTS idx_indicator_ticker_date ON indicator_data(ticker, date);
CREATE INDEX IF NOT EXISTS idx_fundamental_ticker_date ON fundamental_data(ticker, date);
