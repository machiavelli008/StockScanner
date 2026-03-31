from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import numpy as np
import threading
import time
import os

# Ленивый импорт yfinance - только когда нужен
def get_yfinance():
    import yfinance as yf
    return yf

app = FastAPI()

# Health check - простой эндпоинт для проверки, что app работает
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}

frontend_path = str(Path(__file__).parent.parent / "frontend")
print(f"Frontend path: {frontend_path}")

# Глобальный кэш для сигналов
signals_cache = {'signals': [], 'last_update': None}
cache_lock = threading.Lock()
CACHE_EXPIRY_MINUTES = 5  # Кэш устаревает через 5 минут
AUTO_REFRESH_INTERVAL_SECONDS = 300  # Автообновление каждые 5 минут
background_thread_stop = False  # Флаг для остановки фонового потока
# На Vercel отключаем startup refresh (таймаут), но включаем по требованию
ENABLE_STARTUP_REFRESH = os.getenv("ENABLE_STARTUP_REFRESH", "0") == "1" if not os.getenv("VERCEL") else False
ENABLE_BACKGROUND_REFRESH = os.getenv("ENABLE_BACKGROUND_REFRESH", "1") == "1" if not os.getenv("VERCEL") else False
SERVER_PORT = int(os.getenv("PORT", "8001"))

DEFAULT_TICKERS = ["MSFT", "AAPL", "GOOGL", "TSLA", "AMZN"]
TICKERS_FILE_PATH = Path(__file__).parent.parent / "tickers.csv"
SIGNALS_JSON_PATH = Path(__file__).parent.parent / "data" / "signals.json"


def normalize_ohlc_columns(df):
    """Приводит MultiIndex-колонки yfinance к обычным OHLCV колонкам."""
    if isinstance(df.columns, pd.MultiIndex):
        if 'Price' in (df.columns.names or []):
            df.columns = df.columns.get_level_values('Price')
        else:
            df.columns = df.columns.get_level_values(0)
    return df


def load_tickers():
    """Загружает тикеры из tickers.csv, fallback на DEFAULT_TICKERS."""
    if not TICKERS_FILE_PATH.exists():
        print(f"tickers.csv not found: {TICKERS_FILE_PATH}. Using default tickers.")
        return DEFAULT_TICKERS.copy()

    try:
        df = pd.read_csv(TICKERS_FILE_PATH)
        if df.empty:
            print("tickers.csv is empty. Using default tickers.")
            return DEFAULT_TICKERS.copy()

        # Поддерживаем типичные названия колонки: Ticker/ticker/symbol.
        normalized = {str(col).strip().lower(): col for col in df.columns}
        ticker_col = None
        for candidate in ["ticker", "symbol"]:
            if candidate in normalized:
                ticker_col = normalized[candidate]
                break

        if ticker_col is None:
            # Если заголовок нестандартный, берем первую колонку.
            ticker_col = df.columns[0]

        tickers = []
        seen = set()
        for raw in df[ticker_col].dropna().tolist():
            t = str(raw).strip().upper()
            if not t or t in seen:
                continue
            seen.add(t)
            tickers.append(t)

        if not tickers:
            print("No valid tickers in tickers.csv. Using default tickers.")
            return DEFAULT_TICKERS.copy()

        print(f"Loaded {len(tickers)} tickers from tickers.csv")
        return tickers

    except Exception as e:
        print(f"Failed to load tickers.csv: {e}. Using default tickers.")
        return DEFAULT_TICKERS.copy()

def calculate_atr(data, period=14):
    """Расчет ATR"""
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
    """Расчет EMA"""
    return prices.ewm(span=period, adjust=False).mean()

def find_touch_events(
    data,
    ema_col,
    atr_col,
    lookahead=12,
    near_pct=0.0015,
    rebound_pct=0.01,
    lower_ema_cols=None,
    cooldown_bars=0,
):
    """
    Логика касаний сверху вниз:
    - Считаем касание только когда:
        1. Close подходит к EMA в пределах 0.15% сверху (near_from_above), ИЛИ
        2. Low свечи фактически достигает EMA или уходит ниже (actual_touch).
    - Фильтр: если в ту же свечу low достиг более низкой скользящей (longer period)
      — касание текущей EMA НЕ считается (это взаимодействие с более дальней MA).
    - Группа свечей в зоне EMA±ATR = одно событие.
    - Positive: после касания high >= EMA+1%.
    - Negative: close < EMA-1ATR.
    - Минимум 2 свечи для подтверждения результата.
    """

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
        except Exception:
            continue

        if pd.isna(prev_ema) or pd.isna(ema) or pd.isna(atr) or atr <= 0:
            continue

        # Только движение сверху вниз: предыдущая свеча должна быть выше EMA.
        if prev_close <= prev_ema:
            continue

        # Close подходит к EMA в пределах 0.15% оставаясь выше неё.
        near_from_above = ema <= curr_close <= ema * (1 + near_pct)
        # Low фактически достигает уровня EMA или уходит ниже — это реальное касание.
        actual_touch = curr_low <= ema

        if not (near_from_above or actual_touch):
            continue

        # Если уже на стартовой свече задели более низкую MA,
        # считаем это медвежьим признаком (для классификации события).
        touched_lower_ma_on_start = False
        if lower_ema_cols:
            for lower_col in lower_ema_cols:
                try:
                    lower_val = float(data[lower_col].iloc[i])
                except Exception:
                    continue
                if pd.isna(lower_val):
                    continue
                if lower_val < ema and curr_low <= lower_val:
                    touched_lower_ma_on_start = True
                    break

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
            except Exception:
                break

            if pd.isna(f_ema) or pd.isna(f_atr) or f_atr <= 0:
                break

            candles_in_event += 1
            event_indices.append(j)

            # Фиксируем, достигла ли цена более низкой MA в рамках события.
            if lower_ema_cols:
                for lower_col in lower_ema_cols:
                    try:
                        future_lower_val = float(data[lower_col].iloc[j])
                    except Exception:
                        continue
                    if pd.isna(future_lower_val):
                        continue
                    if future_lower_val < f_ema and f_low <= future_lower_val:
                        saw_lower_ma = True
                        break

            # Negative при полном пробое вниз EMA-1ATR.
            # Также считаем Negative, если дошли до более низкой MA.
            if f_close < (f_ema - f_atr) or saw_lower_ma:
                result = 'negative'
                break

            # Positive при реакции вверх минимум +1% от EMA.
            if f_high >= (f_ema * (1 + rebound_pct)) and not saw_lower_ma:
                result = 'positive'
                break

            # Пока цена в зоне EMA±ATR, считаем это одной группой свечей.
            in_consolidation_zone = (f_low <= (f_ema + f_atr) and f_high >= (f_ema - f_atr))
            if not in_consolidation_zone:
                break

        if candles_in_event >= 2 and result in ('positive', 'negative'):
            # Анти-дубль для частых POSITIVE касаний в одной волне.
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


def split_by_year_windows(df, end_date=None):
    """Разделяет данные на окна: 1-5 лет и все 10 лет целиком."""
    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    if end_date is None:
        end_date = out.index.max()

    boundary_5y = end_date - pd.DateOffset(years=5)

    df_1_5y = out[out.index > boundary_5y].copy()
    df_10y = out.copy()  # все 10 лет

    return df_1_5y, df_10y

def get_stock_signals(ticker):
    """Получает сигналы для акции"""
    try:
        print(f"\nDownloading data for {ticker}...")
        
        yf = get_yfinance()
        hist_daily = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=False)
        hist_weekly = yf.download(ticker, period="10y", interval="1wk", progress=False, auto_adjust=False)

        hist_daily = normalize_ohlc_columns(hist_daily)
        hist_weekly = normalize_ohlc_columns(hist_weekly)
        
        if hist_daily.empty or hist_weekly.empty:
            print(f"No data for {ticker}")
            return None
        
        print(f"Daily: {len(hist_daily)} rows, Weekly: {len(hist_weekly)} rows")
        print(f"Calculating indicators for {ticker}...")
        
        current_price = hist_daily['Close'].iloc[-1].item()
        
        ema_periods = [20, 50, 100, 200]
        
        # Daily
        for period in ema_periods:
            hist_daily[f'ema_{period}'] = calculate_ema(hist_daily['Close'], period)
        hist_daily = calculate_atr(hist_daily, 14)
        hist_daily = hist_daily.dropna()
        
        # Weekly
        for period in ema_periods:
            hist_weekly[f'ema_{period}'] = calculate_ema(hist_weekly['Close'], period)
        hist_weekly = calculate_atr(hist_weekly, 14)
        hist_weekly = hist_weekly.dropna()
        
        # Разделяем: 1-5 лет и все 10 лет целиком.
        daily_1_5y, daily_10y = split_by_year_windows(hist_daily)
        weekly_1_5y, weekly_10y = split_by_year_windows(hist_weekly)
        
        print(f"Analyzing touches for {ticker}...")
        
        # Текущие значения EMA и дистанция цены (по дневным данным)
        current_ema = {}
        for period in ema_periods:
            col = f'ema_{period}'
            ema_val = float(hist_daily[col].iloc[-1])
            dist_pct = round(abs(current_price - ema_val) / ema_val * 100, 2)
            current_ema[col] = {
                'value': round(ema_val, 2),
                'distance_pct': dist_pct,
                'is_near': dist_pct <= 1.0,
                'price_above': current_price >= ema_val,
            }

        result = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "current_ema": current_ema,
            "daily": {
                "period_1_5y": {},
                "period_10y": {}
            },
            "weekly": {
                "period_1_5y": {},
                "period_10y": {}
            }
        }
        
        def calc_stats(touches):
            total = len(touches)
            if total == 0:
                return {'positive': 0, 'negative': 0, 'total': 0, 'probability': 0}
            
            positive = len([t for t in touches if t['result'] == 'positive'])
            negative = len([t for t in touches if t['result'] == 'negative'])
            prob = int(round((positive / total) * 100)) if total > 0 else 0
            
            return {
                'positive': positive,
                'negative': negative,
                'total': total,
                'probability': prob
            }
        
        daily_periods = {
            'period_1_5y': daily_1_5y,
            'period_10y': daily_10y
        }
        weekly_periods = {
            'period_1_5y': weekly_1_5y,
            'period_10y': weekly_10y
        }

        # Анализ Daily
        all_ema_cols = [f'ema_{p}' for p in ema_periods]
        for period in ema_periods:
            ema_col = f'ema_{period}'
            lower_emas = [col for col in all_ema_cols if col != ema_col]
            for period_name, period_df in daily_periods.items():
                touches = find_touch_events(
                    period_df,
                    ema_col,
                    'atr',
                    lower_ema_cols=lower_emas,
                    cooldown_bars=0,
                )
                stats = calc_stats(touches)
                result['daily'][period_name][ema_col] = stats

            p1 = result['daily']['period_1_5y'][ema_col]
            p2 = result['daily']['period_10y'][ema_col]
            print(
                f"  Daily EMA{period}: "
                f"1-5y {p1['positive']}/{p1['total']}={p1['probability']}%, "
                f"10y {p2['positive']}/{p2['total']}={p2['probability']}%"
            )
        
        # Анализ Weekly
        for period in ema_periods:
            ema_col = f'ema_{period}'
            lower_emas = [col for col in all_ema_cols if col != ema_col]
            for period_name, period_df in weekly_periods.items():
                touches = find_touch_events(
                    period_df,
                    ema_col,
                    'atr',
                    lower_ema_cols=lower_emas,
                    cooldown_bars=8,
                )
                stats = calc_stats(touches)
                result['weekly'][period_name][ema_col] = stats

            p1 = result['weekly']['period_1_5y'][ema_col]
            p2 = result['weekly']['period_10y'][ema_col]
            print(
                f"  Weekly EMA{period}: "
                f"1-5y {p1['positive']}/{p1['total']}={p1['probability']}%, "
                f"10y {p2['positive']}/{p2['total']}={p2['probability']}%"
            )
        
        print(f"OK - {ticker} completed!")
        return result
        
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def refresh_signals():
    """Обновить кэш сигналов"""
    global signals_cache

    tickers = load_tickers()
    signals = []

    print("\n=== Starting analysis ===")
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(3)  # пауза между тикерами чтобы не получить rate limit
        # до 3 попыток при rate limit
        for attempt in range(3):
            signal = get_stock_signals(ticker)
            if signal is not None:
                signals.append(signal)
                break
            if attempt < 2:
                print(f"Retrying {ticker} in 10s... (attempt {attempt + 2}/3)")
                time.sleep(10)
    
    print(f"\n=== Analysis complete! Found {len(signals)} signals ===\n")
    
    with cache_lock:
        signals_cache['signals'] = signals
        signals_cache['last_update'] = pd.Timestamp.now()


def load_signals_from_file():
    """Загружает сигналы из data/signals.json если файл существует."""
    if not SIGNALS_JSON_PATH.exists():
        return False
    try:
        import json
        with open(SIGNALS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        signals = data.get("signals", [])
        last_update_str = data.get("last_update")
        last_update = pd.Timestamp(last_update_str) if last_update_str else pd.Timestamp.now()
        with cache_lock:
            signals_cache['signals'] = signals
            signals_cache['last_update'] = last_update
        print(f"Loaded {len(signals)} signals from {SIGNALS_JSON_PATH}")
        return True
    except Exception as e:
        print(f"Failed to load signals.json: {e}")
        return False


def should_refresh_cache():
    """Проверяет, нужно ли обновлять кэш сигналов."""
    with cache_lock:
        if not signals_cache['signals']:
            return True
        if signals_cache['last_update'] is None:
            return True

        time_since_update = pd.Timestamp.now() - signals_cache['last_update']
        return time_since_update.total_seconds() > CACHE_EXPIRY_MINUTES * 60

@app.get("/api/signals")
def get_signals():
    """Возвращает сигналы: сначала из файла, иначе из кэша или пересчитывает."""
    if should_refresh_cache():
        loaded = load_signals_from_file()
        if not loaded:
            refresh_signals()

    with cache_lock:
        last_update = signals_cache['last_update']
        last_update_iso = last_update.isoformat() if last_update is not None else None
        return {
            "signals": signals_cache['signals'],
            "last_update": last_update_iso,
        }

@app.post("/api/refresh")
def refresh_signals_endpoint():
    """Обновить сигналы: сначала из файла, иначе через yfinance."""
    loaded = load_signals_from_file()
    if not loaded:
        refresh_signals()
    with cache_lock:
        last_update = signals_cache['last_update']
        last_update_iso = last_update.isoformat() if last_update is not None else None
        return {
            "status": "done",
            "count": len(signals_cache['signals']),
            "last_update": last_update_iso,
        }

@app.get("/api/signals/{ticker}")
def get_signal_by_ticker(ticker: str):
    """Возвращает сигналы для конкретной акции"""
    signal = get_stock_signals(ticker.upper())
    if signal:
        return signal
    return {"error": f"Could not get data for {ticker}"}

# Не монтируем статику на Vercel (она обслуживается отдельно через public/)
# На локальной машине статика монтируется если это не serverless
if not os.getenv("VERCEL"):
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
    except Exception as e:
        print(f"⚠️  Could not mount static files: {e}")

@app.on_event("startup")
async def startup_event():
    """Загружаем сигналы при старте (только локально, не на Vercel)"""
    if ENABLE_STARTUP_REFRESH:
        try:
            print("\n=== Loading signals on startup ===")
            refresh_signals()
        except Exception as e:
            print(f"\n⚠️  Startup refresh failed: {e}")
            print("App will fetch signals on first API call")
    else:
        print("\n=== Startup refresh is disabled ===")
        if os.getenv("VERCEL"):
            print("(Running on Vercel - startup refresh disabled to prevent timeout)")

if __name__ == "__main__":
    # Этот блок выполняется ТОЛЬКО локально, не на Vercel
    import uvicorn
    
    def auto_refresh_background():
        """Фоновый поток для автоматического обновления данных каждые 5 минут"""
        global background_thread_stop
        print("[AUTO-REFRESH] Background updater started. Will refresh every 5 minutes.")
        
        while not background_thread_stop:
            try:
                time.sleep(AUTO_REFRESH_INTERVAL_SECONDS)
                if not background_thread_stop:
                    print(f"[AUTO-REFRESH] Starting automatic refresh at {pd.Timestamp.now()}")
                    refresh_signals()
                    print(f"[AUTO-REFRESH] Automatic refresh completed at {pd.Timestamp.now()}")
            except Exception as e:
                print(f"[AUTO-REFRESH] Error during automatic refresh: {e}")
    
    if ENABLE_BACKGROUND_REFRESH:
        # Запускаем фоновый поток обновления
        bg_thread = threading.Thread(target=auto_refresh_background, daemon=True)
        bg_thread.start()
    else:
        print("[AUTO-REFRESH] Background updater disabled by ENABLE_BACKGROUND_REFRESH")
    
    print("\n" + "="*50)
    print("🚀 StockScanner Backend Starting...")
    print("📊 Data auto-refresh: ENABLED (every 5 minutes)")
    print(f"🌐 API: http://127.0.0.1:{SERVER_PORT}")
    print(f"📄 Docs: http://127.0.0.1:{SERVER_PORT}/docs")
    print("="*50 + "\n")

    try:
        uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
    except OSError as e:
        if "10048" in str(e) or "Address already in use" in str(e):
            fallback_port = SERVER_PORT + 1
            print(f"\n⚠️  Port {SERVER_PORT} already in use. Trying port {fallback_port}...")
            uvicorn.run(app, host="127.0.0.1", port=fallback_port)
        else:
            raise