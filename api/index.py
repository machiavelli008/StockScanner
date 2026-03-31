import os
import threading
import time
from pathlib import Path
from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI()

# Health check - немедленно возвращаем OK
@app.get("/health")
def health():
    return {"status": "ok", "service": "StockScanner API"}

# Глобальный кэш для сигналов
signals_cache = {'signals': [], 'last_update': None}
cache_lock = threading.Lock()
CACHE_EXPIRY_MINUTES = 5

def normalize_ohlc_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        if 'Price' in (df.columns.names or []):
            df.columns = df.columns.get_level_values('Price')
        else:
            df.columns = df.columns.get_level_values(0)
    return df

def calculate_atr(data, period=14):
    data['tr'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(
            abs(data['High'] - data['Close'].shift()),
            abs(data['Low'] - data['Close'].shift())
        )
    )
    data['atr'] = data['tr'].rolling(window=period).mean()
    return data

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def find_touch_events(data, ema_col, atr_col, lookahead=12, near_pct=0.0015, rebound_pct=0.01, lower_ema_cols=None, cooldown_bars=0):
    touches = []
    processed_indices = set()
    last_positive_event_start_idx = -10_000

    for i in range(1, len(data) - 1):
        if i in processed_indices:
            continue
        try:
            prev_close = float(data['Close'].iloc[i - 1])
            prev_ema = float(data[ema_col].iloc[i - 1])
            curr_close = float(data['Close'].iloc[i])
            curr_high = float(data['High'].iloc[i])
            curr_low = float(data['Low'].iloc[i])
            ema = float(data[ema_col].iloc[i])
            atr = float(data[atr_col].iloc[i])
        except:
            continue

        if pd.isna(prev_ema) or pd.isna(ema) or pd.isna(atr) or atr <= 0:
            continue
        if prev_close <= prev_ema:
            continue

        near_from_above = ema <= curr_close <= ema * (1 + near_pct)
        actual_touch = curr_low <= ema

        if not (near_from_above or actual_touch):
            continue

        touched_lower_ma_on_start = False
        if lower_ema_cols:
            for lower_col in lower_ema_cols:
                try:
                    lower_val = float(data[lower_col].iloc[i])
                    if pd.isna(lower_val):
                        continue
                    if lower_val < ema and curr_low <= lower_val:
                        touched_lower_ma_on_start = True
                        break
                except:
                    continue

        event_indices = [i]
        candles_in_event = 1
        result = None
        saw_lower_ma = touched_lower_ma_on_start

        end_j = min(i + lookahead, len(data) - 1)
        for j in range(i + 1, end_j + 1):
            try:
                f_close = float(data['Close'].iloc[j])
                f_high = float(data['High'].iloc[j])
                f_low = float(data['Low'].iloc[j])
                f_ema = float(data[ema_col].iloc[j])
                f_atr = float(data[atr_col].iloc[j])
            except:
                break

            if pd.isna(f_ema) or pd.isna(f_atr) or f_atr <= 0:
                break

            candles_in_event += 1
            event_indices.append(j)

            if lower_ema_cols:
                for lower_col in lower_ema_cols:
                    try:
                        future_lower_val = float(data[lower_col].iloc[j])
                        if not pd.isna(future_lower_val) and future_lower_val < f_ema and f_low <= future_lower_val:
                            saw_lower_ma = True
                            break
                    except:
                        continue

            if f_close < (f_ema - f_atr) or saw_lower_ma:
                result = 'negative'
                break
            if f_high >= (f_ema * (1 + rebound_pct)) and not saw_lower_ma:
                result = 'positive'
                break

            in_consolidation_zone = (f_low <= (f_ema + f_atr) and f_high >= (f_ema - f_atr))
            if not in_consolidation_zone:
                break

        if candles_in_event >= 2 and result in ('positive', 'negative'):
            if result == 'positive' and (i - last_positive_event_start_idx <= cooldown_bars):
                for idx in event_indices:
                    processed_indices.add(idx)
                continue

            touches.append({
                'index': i,
                'date': data.index[i],
                'price': curr_close,
                'ema': ema,
                'atr': atr,
                'result': result
            })
            if result == 'positive':
                last_positive_event_start_idx = i

        for idx in event_indices:
            processed_indices.add(idx)

    return touches

def get_stock_signals(ticker):
    try:
        print(f"Downloading data for {ticker}...")
        import yfinance as yf
        
        hist_daily = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=False)
        hist_weekly = yf.download(ticker, period="10y", interval="1wk", progress=False, auto_adjust=False)

        hist_daily = normalize_ohlc_columns(hist_daily)
        hist_weekly = normalize_ohlc_columns(hist_weekly)
        
        if hist_daily.empty or hist_weekly.empty:
            return None
        
        current_price = hist_daily['Close'].iloc[-1].item()
        ema_periods = [20, 50, 100, 200]
        
        for period in ema_periods:
            hist_daily[f'ema_{period}'] = calculate_ema(hist_daily['Close'], period)
        hist_daily = calculate_atr(hist_daily, 14)
        hist_daily = hist_daily.dropna()
        
        for period in ema_periods:
            hist_weekly[f'ema_{period}'] = calculate_ema(hist_weekly['Close'], period)
        hist_weekly = calculate_atr(hist_weekly, 14)
        hist_weekly = hist_weekly.dropna()
        
        daily_1_5y, daily_10y = hist_daily[hist_daily.index > hist_daily.index[-1] - pd.DateOffset(years=5)].copy(), hist_daily.copy()
        weekly_1_5y, weekly_10y = hist_weekly[hist_weekly.index > hist_weekly.index[-1] - pd.DateOffset(years=5)].copy(), hist_weekly.copy()
        
        result = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "daily": {"period_1_5y": {}, "period_10y": {}},
            "weekly": {"period_1_5y": {}, "period_10y": {}}
        }
        
        def calc_stats(touches):
            total = len(touches)
            if total == 0:
                return {'positive': 0, 'negative': 0, 'total': 0, 'probability': 0}
            positive = len([t for t in touches if t['result'] == 'positive'])
            negative = len([t for t in touches if t['result'] == 'negative'])
            prob = int(round((positive / total) * 100)) if total > 0 else 0
            return {'positive': positive, 'negative': negative, 'total': total, 'probability': prob}
        
        all_ema_cols = [f'ema_{p}' for p in ema_periods]
        for period in ema_periods:
            ema_col = f'ema_{period}'
            lower_emas = [col for col in all_ema_cols if col != ema_col]
            
            for df, key in [(daily_1_5y, 'period_1_5y'), (daily_10y, 'period_10y')]:
                touches = find_touch_events(df, ema_col, 'atr', lower_ema_cols=lower_emas, cooldown_bars=0)
                result['daily'][key][ema_col] = calc_stats(touches)
            
            for df, key in [(weekly_1_5y, 'period_1_5y'), (weekly_10y, 'period_10y')]:
                touches = find_touch_events(df, ema_col, 'atr', lower_ema_cols=lower_emas, cooldown_bars=8)
                result['weekly'][key][ema_col] = calc_stats(touches)
        
        return result
        
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return None

def refresh_signals():
    global signals_cache
    DEFAULT_TICKERS = ["MSFT", "AAPL", "GOOGL", "TSLA", "AMZN"]
    signals = []
    
    for ticker in DEFAULT_TICKERS:
        signal = get_stock_signals(ticker)
        if signal:
            signals.append(signal)
    
    with cache_lock:
        signals_cache['signals'] = signals
        signals_cache['last_update'] = pd.Timestamp.now()

def should_refresh_cache():
    with cache_lock:
        if not signals_cache['signals']:
            return True
        if signals_cache['last_update'] is None:
            return True
        time_since_update = pd.Timestamp.now() - signals_cache['last_update']
        return time_since_update.total_seconds() > CACHE_EXPIRY_MINUTES * 60

@app.get("/api/signals")
def get_signals():
    if should_refresh_cache():
        refresh_signals()
    with cache_lock:
        last_update = signals_cache['last_update']
        last_update_iso = last_update.isoformat() if last_update is not None else None
        return {"signals": signals_cache['signals'], "last_update": last_update_iso}

@app.post("/api/refresh")
def refresh_signals_endpoint():
    refresh_signals()
    with cache_lock:
        last_update = signals_cache['last_update']
        last_update_iso = last_update.isoformat() if last_update is not None else None
        return {"status": "done", "count": len(signals_cache['signals']), "last_update": last_update_iso}

@app.get("/api/signals/{ticker}")
def get_signal_by_ticker(ticker: str):
    signal = get_stock_signals(ticker.upper())
    if signal:
        return signal
    return {"error": f"Could not get data for {ticker}"}



