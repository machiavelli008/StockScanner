from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import time
import os
import io
import json

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
    """Загружает тикеры из tickers.csv, возвращает список (ticker, category).
    Fallback на DEFAULT_TICKERS с категорией 'Other'."""
    if not TICKERS_FILE_PATH.exists():
        print(f"tickers.csv not found: {TICKERS_FILE_PATH}. Using default tickers.")
        return [(t, 'Other') for t in DEFAULT_TICKERS]

    try:
        df = pd.read_csv(TICKERS_FILE_PATH)
        if df.empty:
            print("tickers.csv is empty. Using default tickers.")
            return [(t, 'Other') for t in DEFAULT_TICKERS]

        # Поддерживаем типичные названия колонки: Ticker/ticker/symbol.
        normalized = {str(col).strip().lower(): col for col in df.columns}
        ticker_col = None
        for candidate in ["ticker", "symbol"]:
            if candidate in normalized:
                ticker_col = normalized[candidate]
                break

        if ticker_col is None:
            ticker_col = df.columns[0]

        # Колонка категории
        category_col = normalized.get("category")

        result = []
        seen = set()
        for _, row in df.iterrows():
            raw = row[ticker_col]
            if pd.isna(raw):
                continue
            t = str(raw).strip().upper()
            if not t or t in seen:
                continue
            seen.add(t)
            cat = 'Other'
            if category_col:
                cat_val = row[category_col]
                if not pd.isna(cat_val):
                    cat = str(cat_val).strip()
            result.append((t, cat))

        if not result:
            print("No valid tickers in tickers.csv. Using default tickers.")
            return [(t, 'Other') for t in DEFAULT_TICKERS]

        print(f"Loaded {len(result)} tickers from tickers.csv")
        return result

    except Exception as e:
        print(f"Failed to load tickers.csv: {e}. Using default tickers.")
        return [(t, 'Other') for t in DEFAULT_TICKERS]

def compute_current_ema_signals(hist, current_price, ema_periods):
    """Считает signal_type для каждой EMA на основе текущей цены и данных таймфрейма."""
    current_atr = float(hist['atr'].iloc[-1])
    ema_values = {
        f'ema_{p}': float(hist[f'ema_{p}'].iloc[-1])
        for p in ema_periods
    }
    result = {}
    for period in ema_periods:
        col = f'ema_{period}'
        ema_val = ema_values[col]
        dist_pct = round(abs(current_price - ema_val) / ema_val * 100, 2)
        price_above = current_price >= ema_val
        signal_type = None

        if price_above:
            # Проверяем что последние 3 свечи тоже были выше EMA (цена снижается сверху вниз)
            prev_closes = [float(hist['Close'].iloc[-i]) for i in range(2, 5)]
            prev_emas = [float(hist[col].iloc[-i]) for i in range(2, 5)]
            came_from_above = all(c >= e for c, e in zip(prev_closes, prev_emas))
            if came_from_above:
                if dist_pct <= 0.15:
                    signal_type = 'entry_zone'
                elif dist_pct <= 1.0:
                    signal_type = 'approaching'
        else:
            lower_emas = [v for k, v in ema_values.items() if v < ema_val]
            touches_lower = any(current_price <= lv for lv in lower_emas)
            if not touches_lower:
                prev_closes = [float(hist['Close'].iloc[-i]) for i in range(2, 5)]
                prev_emas = [float(hist[col].iloc[-i]) for i in range(2, 5)]
                came_from_above = all(c >= e for c, e in zip(prev_closes, prev_emas))

                if dist_pct <= 0.15:
                    if came_from_above:
                        signal_type = 'entry_zone'  # только что упала — касание
                    elif period == 200:
                        signal_type = 'watching'    # EMA200: подход снизу к уровню
                elif dist_pct <= 2.0:
                    if came_from_above:
                        signal_type = 'watching'    # только что упала ниже EMA
                    elif period == 200:
                        signal_type = 'watching'    # EMA200: подход снизу, консолидация

        # EMA20: только в восходящем тренде — EMA20 и EMA50 обе должны расти
        if period == 20 and signal_type is not None and len(hist) > 20:
            ema20_past = float(hist[col].iloc[-21])
            ema50_now = float(hist['ema_50'].iloc[-1])
            ema50_past = float(hist['ema_50'].iloc[-21])
            ema20_rising = ema_val > ema20_past
            ema50_rising = ema50_now > ema50_past
            if not (ema20_rising and ema50_rising):
                signal_type = None

        # EMA20: подавляем если текущая цена уже у EMA50 (цена упала слишком глубоко)
        if period == 20 and signal_type is not None:
            ema50_val = float(hist['ema_50'].iloc[-1])
            if current_price <= ema50_val * 1.01:
                signal_type = None

        result[col] = {
            'value': round(ema_val, 2),
            'distance_pct': dist_pct,
            'price_above': price_above,
            'signal_type': signal_type,
        }
    return result


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
    require_rally_after_negative=False,
    max_bars_below_ema=None,
    min_ema_slope_bars=0,
    min_ema_slope_pct=0.015,
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
    last_negative_event_end_idx = -10_000

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

        # Только движение сверху вниз: минимум 3 свечи подряд были выше EMA перед касанием.
        came_from_above = all(
            float(data['Close'].iloc[i - k]) > float(data[ema_col].iloc[i - k])
            for k in range(1, 4)
            if i - k >= 0
        )
        if not came_from_above:
            continue

        # Фильтр наклона EMA: EMA должна вырасти минимум на min_ema_slope_pct за N баров назад.
        # Если EMA flat или почти не движется — боковик/нисходящий тренд, касание не считаем.
        if min_ema_slope_bars > 0 and i >= min_ema_slope_bars:
            ema_past = float(data[ema_col].iloc[i - min_ema_slope_bars])
            if not pd.isna(ema_past) and ema_past > 0:
                slope_pct = (ema - ema_past) / ema_past
                if slope_pct < min_ema_slope_pct:
                    continue

        # Close подходит к EMA в пределах 0.15% оставаясь выше неё.
        near_from_above = ema <= curr_close <= ema * (1 + near_pct)
        # Low фактически достигает уровня EMA или уходит ниже — это реальное касание.
        actual_touch = curr_low <= ema

        if not (near_from_above or actual_touch):
            continue

        # Если уже на стартовой свече задели более низкую MA,
        # считаем это медвежьим признаком (для классификации события).
        # Если на стартовой свече LOW уже задел более низкую EMA —
        # касание не считаем вообще: непонятно на какую EMA реагирует цена.
        if lower_ema_cols:
            touched_lower_ma_on_start = False
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
            if touched_lower_ma_on_start:
                processed_indices.add(i)
                continue

        event_indices = [i]
        candles_in_event = 1
        result = None
        bars_below_ema = 1 if float(data['Close'].iloc[i]) < float(data[ema_col].iloc[i]) else 0
        touched_lower_ema_during = False
        max_high_seen = curr_high

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
            max_high_seen = max(max_high_seen, f_high)

            # Если во время события LOW коснулся более низкой EMA — запоминаем флаг
            if lower_ema_cols and not touched_lower_ema_during:
                for lower_col in lower_ema_cols:
                    try:
                        lower_val = float(data[lower_col].iloc[j])
                    except Exception:
                        continue
                    if not pd.isna(lower_val) and lower_val < f_ema and f_low <= lower_val:
                        touched_lower_ema_during = True
                        break

            # Счётчик баров ниже EMA
            if f_close < f_ema:
                bars_below_ema += 1
            else:
                bars_below_ema = 0

            # Негатив если цена закрылась ниже EMA N баров подряд (только weekly)
            if max_bars_below_ema and bars_below_ema >= max_bars_below_ema:
                result = 'negative'
                break

            # Positive: High достиг EMA + 1 ATR (цена отскочила на 1 ATR от уровня).
            # Если во время события LOW касался нижней EMA — позитив не считаем (неясно на что реакция).
            # Для weekly: требуем подтверждение — следующие 3 бара не должны закрыться ниже EMA−ATR.
            if f_high >= (f_ema + f_atr):
                if touched_lower_ema_during:
                    result = None
                    break
                if require_rally_after_negative:  # применяем только для weekly
                    confirmed = True
                    for k in range(1, 4):
                        ki = j + k
                        if ki >= len(data):
                            break
                        try:
                            kc = float(data['Close'].iloc[ki])
                            ke = float(data[ema_col].iloc[ki])
                            ka = float(data[atr_col].iloc[ki])
                        except Exception:
                            break
                        if not (pd.isna(ke) or pd.isna(ka)) and kc < (ke - ka):
                            confirmed = False
                            break
                    result = 'positive' if confirmed else 'negative'
                else:
                    result = 'positive'
                break

            # Negative: закрылась ниже EMA − 1ATR.
            if f_close < (f_ema - f_atr):
                result = 'negative'
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

            # Фильтр боковика (только weekly): после негативного события требуем ралли —
            # хотя бы одно закрытие выше EMA+ATR. Если его не было — цена в боковике, не считаем.
            if require_rally_after_negative and last_negative_event_end_idx > -10_000:
                had_rally = any(
                    float(data['Close'].iloc[k]) >= float(data[ema_col].iloc[k]) + float(data[atr_col].iloc[k])
                    for k in range(last_negative_event_end_idx + 1, i)
                    if k >= 0
                )
                if not had_rally:
                    for idx in event_indices:
                        processed_indices.add(idx)
                    continue

            max_rally_pct = round((max_high_seen - curr_close) / curr_close * 100, 2) if result == 'positive' else None
            touches.append({
                'index': i,
                'date': data.index[i],
                'price': curr_close,
                'ema': ema,
                'atr': atr,
                'result': result,
                'max_rally_pct': max_rally_pct,
            })
            if result == 'positive':
                last_positive_event_start_idx = i
            elif result == 'negative':
                last_negative_event_end_idx = event_indices[-1]

        for idx in event_indices:
            processed_indices.add(idx)

    return touches


def detect_weekly_hammers(hist_weekly, lookback=2):
    """
    Ищет молоты на недельном таймфрейме в последних lookback свечах.
    Молот: нижняя тень >= 60%, тело <= 30%, верхняя тень <= 20% всей свечи.
    Возвращает dict: {'count': int, 'dates': [str]} или None если молотов нет.
    """
    if len(hist_weekly) < lookback:
        return None

    hammers = []
    for i in range(-lookback, 0):
        try:
            o = float(hist_weekly['Open'].iloc[i])
            h = float(hist_weekly['High'].iloc[i])
            l = float(hist_weekly['Low'].iloc[i])
            c = float(hist_weekly['Close'].iloc[i])
        except Exception:
            continue

        candle_range = h - l
        if candle_range <= 0:
            continue

        body = abs(c - o)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l

        body_pct        = body / candle_range
        upper_shadow_pct = upper_shadow / candle_range
        lower_shadow_pct = lower_shadow / candle_range

        if not (lower_shadow_pct >= 0.60 and body_pct <= 0.30 and upper_shadow_pct <= 0.20):
            continue

        # Молот засчитывается только если закрытие на уровне EMA20 или ниже
        try:
            ema20 = float(hist_weekly['ema_20'].iloc[i])
        except Exception:
            continue
        if c > ema20:
            continue

        date_str = str(hist_weekly.index[i])[:10]
        hammers.append(date_str)

    if not hammers:
        return None

    return {'count': len(hammers), 'dates': hammers}


def compute_rally_between_touches(touches, data, cluster_bars=20):
    """
    Группирует касания в кластеры по времени и считает ралли от первого касания
    кластера до начала следующего кластера.
    Кластер: последовательные позитивные касания в пределах cluster_bars баров.
    """
    pos = [t for t in touches if t['result'] == 'positive']
    for t in touches:
        if t['result'] != 'positive':
            t['max_rally_pct'] = None

    if not pos:
        return touches

    # Строим кластеры по временному расстоянию между касаниями
    clusters = [[pos[0]]]
    for k in range(1, len(pos)):
        prev = pos[k - 1]
        curr = pos[k]
        bars_between = curr['index'] - prev['index']
        if bars_between <= cluster_bars:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])

    # Для каждого кластера: ралли от первого касания до начала следующего кластера
    for c_idx, cluster in enumerate(clusters):
        first_touch = cluster[0]
        start_i     = first_touch['index']
        touch_price = first_touch['price']

        end_i = clusters[c_idx + 1][0]['index'] if c_idx + 1 < len(clusters) else len(data) - 1
        max_high = float(data['High'].iloc[start_i:end_i + 1].max())
        rally_pct = round((max_high - touch_price) / touch_price * 100, 2)

        first_touch['max_rally_pct'] = rally_pct
        for t in cluster[1:]:
            t['max_rally_pct'] = None  # не учитываем в среднем — это один кластер

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

def get_stock_signals(ticker, category='Other'):
    """Получает сигналы для акции"""
    try:
        print(f"\nDownloading data for {ticker}...")
        
        yf = get_yfinance()
        # Скачиваем 15 лет для точного расчёта EMA200, анализируем последние 10
        hist_daily = yf.download(ticker, period="15y", interval="1d", progress=False, auto_adjust=False)
        hist_weekly = yf.download(ticker, period="15y", interval="1wk", progress=False, auto_adjust=False)

        hist_daily = normalize_ohlc_columns(hist_daily)
        hist_weekly = normalize_ohlc_columns(hist_weekly)
        
        if hist_daily.empty or hist_weekly.empty:
            print(f"No data for {ticker}")
            return None
        
        print(f"Daily: {len(hist_daily)} rows, Weekly: {len(hist_weekly)} rows")
        print(f"Calculating indicators for {ticker}...")

        current_price = hist_daily['Close'].iloc[-1].item()

        # Фильтр постоянно падающих инструментов (обратные/плечевые ETF с decay).
        # Если за 5 лет потеряно более 70% — анализ касаний бессмысленен.
        downtrend_warning = None
        hist_daily_idx = hist_daily.copy()
        hist_daily_idx.index = pd.to_datetime(hist_daily_idx.index)
        cutoff_5y = hist_daily_idx.index.max() - pd.DateOffset(years=5)
        hist_5y_start = hist_daily_idx[hist_daily_idx.index >= cutoff_5y]
        if not hist_5y_start.empty:
            price_5y_ago = float(hist_5y_start['Close'].iloc[0])
            if price_5y_ago > 0:
                change_5y = (current_price - price_5y_ago) / price_5y_ago
                if change_5y < -0.70:
                    downtrend_warning = (
                        f"Long-term downtrend detected: -{abs(change_5y)*100:.0f}% over 5 years. "
                        "EMA touch statistics may not be reliable for this instrument."
                    )
                    print(f"  WARNING: {ticker} lost {abs(change_5y)*100:.0f}% in 5Y — flagged as downtrending")
        
        ema_periods = [20, 50, 100, 200]

        # Daily — EMA считаем на 15 годах, потом обрезаем до последних 10 для анализа
        for period in ema_periods:
            hist_daily[f'ema_{period}'] = calculate_ema(hist_daily['Close'], period)
        hist_daily = calculate_atr(hist_daily, 14)
        hist_daily = hist_daily.dropna()
        hist_daily.index = pd.to_datetime(hist_daily.index)
        cutoff_10y = hist_daily.index.max() - pd.DateOffset(years=10)
        hist_daily = hist_daily[hist_daily.index > cutoff_10y]

        # Weekly — то же самое
        for period in ema_periods:
            hist_weekly[f'ema_{period}'] = calculate_ema(hist_weekly['Close'], period)
        # EMA150 только для недельного (для детектора молотов)
        hist_weekly['ema_150'] = calculate_ema(hist_weekly['Close'], 150)
        hist_weekly = calculate_atr(hist_weekly, 14)
        hist_weekly = hist_weekly.dropna()
        hist_weekly.index = pd.to_datetime(hist_weekly.index)
        hist_weekly = hist_weekly[hist_weekly.index > cutoff_10y]

        # Разделяем: 1-5 лет и все 10 лет целиком.
        daily_1_5y, daily_10y = split_by_year_windows(hist_daily)
        weekly_1_5y, weekly_10y = split_by_year_windows(hist_weekly)

        # Определяем боковик: смотрим на EMA50 недельного за 10 лет.
        # Если >50% 8-недельных окон показывают рост EMA50 < 1.5% — сток часто в боковике.
        sideways_warning = None
        slope_bars = 8
        ema50_w = hist_weekly['ema_50'].dropna()
        if len(ema50_w) > slope_bars + 5:
            sideways_count = 0
            total_count = 0
            for idx in range(slope_bars, len(ema50_w)):
                e_now  = float(ema50_w.iloc[idx])
                e_past = float(ema50_w.iloc[idx - slope_bars])
                if e_past > 0:
                    slope = (e_now - e_past) / e_past
                    if slope < 0.015:
                        sideways_count += 1
                    total_count += 1
            if total_count > 0:
                sideways_ratio = sideways_count / total_count
                if sideways_ratio > 0.50:
                    sideways_warning = (
                        f"Frequent sideways periods detected: {int(sideways_ratio*100)}% of time "
                        "EMA50 was flat or declining. Touch statistics may be less reliable."
                    )
                    print(f"  WARNING: {ticker} sideways {int(sideways_ratio*100)}% of time — flagged")

        print(f"Analyzing touches for {ticker}...")

        # Плашки считаются отдельно для дневного и недельного таймфреймов
        current_ema_daily = compute_current_ema_signals(hist_daily, current_price, ema_periods)
        current_ema_weekly = compute_current_ema_signals(hist_weekly, current_price, ema_periods)

        result = {
            "ticker": ticker,
            "category": category,
            "current_price": round(current_price, 2),
            "downtrend_warning": downtrend_warning,
            "sideways_warning": sideways_warning,
            "current_ema": current_ema_daily,
            "current_ema_weekly": current_ema_weekly,
            "hammer_signal": detect_weekly_hammers(hist_weekly),
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
                return {'positive': 0, 'negative': 0, 'total': 0, 'probability': 0, 'avg_rally_pct': None}

            positive = len([t for t in touches if t['result'] == 'positive'])
            negative = len([t for t in touches if t['result'] == 'negative'])
            prob = int(round((positive / total) * 100)) if total > 0 else 0

            rallies = [t['max_rally_pct'] for t in touches if t.get('max_rally_pct') is not None]
            avg_rally_pct = round(sum(rallies) / len(rallies), 1) if rallies else None

            return {
                'positive': positive,
                'negative': negative,
                'total': total,
                'probability': prob,
                'avg_rally_pct': avg_rally_pct,
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
            # EMA20: не фильтруем по lower_ema — если low ушёл ниже EMA50,
            # это негативное касание EMA20 (не удержали), lookahead сам даст negative.
            lower_emas = [] if period == 20 else [col for col in all_ema_cols if col != ema_col]
            for period_name, period_df in daily_periods.items():
                touches = find_touch_events(
                    period_df,
                    ema_col,
                    'atr',
                    lower_ema_cols=lower_emas if lower_emas else None,
                    cooldown_bars=0,
                    min_ema_slope_pct=0.01 if period == 20 else 0.015,
                )
                if period == 200:
                    touches = compute_rally_between_touches(touches, period_df, cluster_bars=20)
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
            lower_emas = [] if period == 20 else [col for col in all_ema_cols if col != ema_col]
            for period_name, period_df in weekly_periods.items():
                touches = find_touch_events(
                    period_df,
                    ema_col,
                    'atr',
                    lower_ema_cols=lower_emas if lower_emas else None,
                    cooldown_bars=2,
                    require_rally_after_negative=True,
                    max_bars_below_ema=3 if period <= 50 else None,
                    min_ema_slope_bars=8 if period <= 50 else 0,
                    min_ema_slope_pct=0.01 if period == 20 else 0.015,
                )
                if period == 200:
                    touches = compute_rally_between_touches(touches, period_df, cluster_bars=26)
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

    ticker_list = load_tickers()
    signals = []

    print("\n=== Starting analysis ===")
    for i, (ticker, category) in enumerate(ticker_list):
        if i > 0:
            time.sleep(3)  # пауза между тикерами чтобы не получить rate limit
        # до 3 попыток при rate limit
        for attempt in range(3):
            signal = get_stock_signals(ticker, category)
            if signal is not None:
                signals.append(signal)
                break
            if attempt < 2:
                print(f"Retrying {ticker} in 10s... (attempt {attempt + 2}/3)")
                time.sleep(10)
    
    print(f"\n=== Analysis complete! Found {len(signals)} signals ===\n")

    now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        SIGNALS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_update": now_str, "signals": signals}, f, ensure_ascii=False, default=str)
        print(f"Saved signals.json (last_update: {now_str})")
    except Exception as e:
        print(f"Failed to save signals.json: {e}")

    with cache_lock:
        signals_cache['signals'] = signals
        signals_cache['last_update'] = pd.Timestamp.now()
        signals_cache['data_generated_at'] = now_str


def load_signals_from_file():
    """Загружает сигналы из data/signals.json если файл существует."""
    if not SIGNALS_JSON_PATH.exists():
        return False
    try:
        import json
        with open(SIGNALS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        signals = data.get("signals", [])
        data_generated_at = data.get("last_update", "unknown")
        with cache_lock:
            signals_cache['signals'] = signals
            signals_cache['last_update'] = pd.Timestamp.now()  # время загрузки на сервере
            signals_cache['data_generated_at'] = data_generated_at
        print(f"Loaded {len(signals)} signals from {SIGNALS_JSON_PATH} (generated: {data_generated_at})")
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
    """Возвращает сигналы из signals.json."""
    if should_refresh_cache():
        load_signals_from_file()

    with cache_lock:
        last_update = signals_cache['last_update']
        last_update_iso = last_update.isoformat() if last_update is not None else datetime.now().isoformat()
        data_generated_at = signals_cache.get('data_generated_at')
        return {
            "signals": signals_cache['signals'],
            "last_update": last_update_iso,
            "data_generated_at": data_generated_at,
        }

@app.post("/api/refresh")
def refresh_signals_endpoint():
    """Запускает пересчёт сигналов в фоне и сразу возвращает ответ."""
    thread = threading.Thread(target=refresh_signals, daemon=True)
    thread.start()
    return {"status": "started"}

@app.get("/api/signals/{ticker}")
def get_signal_by_ticker(ticker: str):
    """Возвращает сигналы для конкретной акции"""
    signal = get_stock_signals(ticker.upper())
    if signal:
        return signal
    return {"error": f"Could not get data for {ticker}"}


def _build_excel_from_rows(rows: list[dict]) -> io.BytesIO:
    """Генерирует Excel из списка строк."""
    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="EMA Analysis")
        ws = writer.sheets["EMA Analysis"]
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) for c in col_cells if c.value), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 25)
    output.seek(0)
    return output


class BatchExcelRequest(BaseModel):
    rows: list[dict]


@app.post("/api/batch-to-excel")
def batch_to_excel(req: BatchExcelRequest):
    """Конвертирует готовые данные в Excel (без повторного запроса к yfinance)."""
    if not req.rows:
        return {"error": "No data"}
    output = _build_excel_from_rows(req.rows)
    filename = f"ema_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@app.get("/api/debug")
def debug_info():
    file_exists = SIGNALS_JSON_PATH.exists()
    import hashlib
    file_hash = hashlib.md5(open(SIGNALS_JSON_PATH, 'rb').read()).hexdigest() if file_exists else None
    cache_count = len(signals_cache['signals'])
    aapl = next((s for s in signals_cache['signals'] if s.get('ticker') == 'AAPL'), None)
    ema200_10y = aapl['weekly']['period_10y']['ema_200'] if aapl else None
    ema20_1_5y = aapl['daily']['period_1_5y']['ema_20'] if aapl else None
    return {
        "signals_json_path": str(SIGNALS_JSON_PATH),
        "file_exists": file_exists,
        "file_md5": file_hash,
        "cache_count": cache_count,
        "last_update": signals_cache['last_update'].isoformat() if signals_cache['last_update'] else None,
        "aapl_daily_ema20_1_5y": ema20_1_5y,
        "aapl_weekly_ema200_10y": ema200_10y,
    }

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
    """Загружаем сигналы при старте"""
    # Всегда пробуем загрузить из файла (быстро, без сети)
    loaded = load_signals_from_file()
    if loaded:
        print("\n=== Signals loaded from file on startup ===")
    elif ENABLE_STARTUP_REFRESH:
        try:
            print("\n=== Loading signals on startup via yfinance ===")
            refresh_signals()
        except Exception as e:
            print(f"\n⚠️  Startup refresh failed: {e}")
            print("App will fetch signals on first API call")
    else:
        print("\n=== No signals file found, will load on first request ===")
        if os.getenv("VERCEL"):
            print("(Running on Vercel - startup refresh disabled to prevent timeout)")

if __name__ == "__main__":
    # Этот блок выполняется ТОЛЬКО локально, не на Vercel
    import uvicorn
    
    def auto_refresh_background():
        """
        Пн-Чт: за 45 мин до закрытия NYSE.
        Пт: за 1 час до закрытия NYSE (покрывает дневные и недельные свечи).
        Летнее время США (EDT, UTC-4): Пн-Чт 19:15 UTC, Пт 19:00 UTC.
        Зимнее время США (EST, UTC-5): Пн-Чт 20:15 UTC, Пт 20:00 UTC.
        """
        import zoneinfo
        global background_thread_stop
        print("[AUTO-REFRESH] Started. Summer: Mon-Thu 19:15 UTC / Fri 19:00 UTC. Winter: Mon-Thu 20:15 UTC / Fri 20:00 UTC.")

        et_tz = zoneinfo.ZoneInfo("America/New_York")

        while not background_thread_stop:
            try:
                now_utc = pd.Timestamp.now('UTC')
                now_et = now_utc.tz_convert(et_tz)
                # Определяем смещение ET от UTC (-4 летом, -5 зимой)
                et_offset = now_et.utcoffset().total_seconds() / 3600  # -4 или -5
                is_friday = now_utc.dayofweek == 4
                # NYSE закрывается в 16:00 ET = 20:00 UTC (зима) или 19:00 UTC (лето)
                # Пн-Чт: за 45 мин до закрытия; Пт: за 1 час до закрытия
                close_utc_hour = 16 + abs(int(et_offset))  # 20 зимой, 19 летом
                refresh_minute = 0 if is_friday else 15
                refresh_utc_hour = close_utc_hour - 1 if is_friday else close_utc_hour
                next_run = now_utc.normalize() + pd.Timedelta(hours=refresh_utc_hour, minutes=refresh_minute)
                if next_run <= now_utc:
                    next_run += pd.Timedelta(days=1)
                while next_run.dayofweek >= 5:  # пропускаем выходные
                    next_run += pd.Timedelta(days=1)
                wait_seconds = (next_run - now_utc).total_seconds()
                print(f"[AUTO-REFRESH] Next: {next_run.strftime('%Y-%m-%d %H:%M UTC')} (in {int(wait_seconds/3600)}h {int((wait_seconds%3600)/60)}m)")
                time.sleep(wait_seconds)
                if not background_thread_stop:
                    print(f"[AUTO-REFRESH] Starting refresh at {pd.Timestamp.now()}")
                    refresh_signals()
                    print(f"[AUTO-REFRESH] Completed at {pd.Timestamp.now()}")
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
    print("📊 Data auto-refresh: Summer Mon-Thu 19:15/Fri 19:00 UTC | Winter Mon-Thu 20:15/Fri 20:00 UTC")
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