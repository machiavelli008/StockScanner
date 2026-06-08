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
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import screener as screener_module
import analyzer as analyzer_module

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
    # При параллельных загрузках yfinance может вернуть дублирующиеся колонки
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_tickers():
    """
    Загружает активный список тикеров.
    Приоритет: active_tickers.json (скринер) → tickers.csv → DEFAULT_TICKERS.
    """
    active_path = screener_module.ACTIVE_TICKERS_PATH
    if active_path.exists():
        try:
            data = json.loads(active_path.read_text(encoding='utf-8'))
            if data:
                result = [(t, d.get('category', 'Screener')) for t, d in data.items()]
                print(f"Loaded {len(result)} tickers from active_tickers.json")
                return result
        except Exception as e:
            print(f"Failed to load active_tickers.json: {e}")

    # Fallback: tickers.csv
    if not TICKERS_FILE_PATH.exists():
        print(f"tickers.csv not found. Using default tickers.")
        return [(t, 'Other') for t in DEFAULT_TICKERS]

    try:
        df = pd.read_csv(TICKERS_FILE_PATH)
        if df.empty:
            return [(t, 'Other') for t in DEFAULT_TICKERS]

        normalized = {str(col).strip().lower(): col for col in df.columns}
        ticker_col = None
        for candidate in ["ticker", "symbol"]:
            if candidate in normalized:
                ticker_col = normalized[candidate]
                break
        if ticker_col is None:
            ticker_col = df.columns[0]
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
            return [(t, 'Other') for t in DEFAULT_TICKERS]

        print(f"Loaded {len(result)} tickers from tickers.csv")
        return result

    except Exception as e:
        print(f"Failed to load tickers.csv: {e}. Using default tickers.")
        return [(t, 'Other') for t in DEFAULT_TICKERS]

def compute_current_ema_signals(hist, current_price, ema_periods, is_weekly=False):
    """Считает signal_type для каждой EMA на основе текущей цены и данных таймфрейма."""
    current_atr = float(hist['atr'].iloc[-1])
    current_low = float(hist['Low'].iloc[-1])
    ema_values = {
        f'ema_{p}': float(hist[f'ema_{p}'].iloc[-1])
        for p in ema_periods
    }

    # Пороги по таймфреймам:
    # Weekly: вход ≤ 2%, approaching 2–4%
    # Daily:  вход ≤ 1%, approaching 1–2%
    entry_pct   = 2.0 if is_weekly else 1.0
    wick_pct    = 2.0 if is_weekly else 1.0
    approach_pct = 4.0 if is_weekly else 2.0

    # EMA10 нужна только для логики EMA20: верхняя граница зоны входа
    ema10_val = float(hist['ema_10'].iloc[-1]) if 'ema_10' in hist.columns else None

    # Глобальный фильтр тренда: цена ниже 200 EMA = нисходящий тренд, сигналы не даём
    ema_200_val = ema_values.get('ema_200')
    if ema_200_val and current_price < ema_200_val:
        return {
            f'ema_{p}': {
                'value': round(ema_values[f'ema_{p}'], 2),
                'atr': round(current_atr, 2),
                'distance_pct': round(abs(current_price - ema_values[f'ema_{p}']) / ema_values[f'ema_{p}'] * 100, 2),
                'price_above': current_price >= ema_values[f'ema_{p}'],
                'signal_type': None,
                'wick_touch': False,
            }
            for p in ema_periods
        }

    result = {}
    for period in ema_periods:
        col = f'ema_{period}'
        ema_val = ema_values[col]
        dist_pct = round(abs(current_price - ema_val) / ema_val * 100, 2)
        price_above = current_price >= ema_val
        low_wick_touch = price_above and current_low <= ema_val * (1 + wick_pct / 100)
        signal_type = None

        prev_closes = [float(hist['Close'].iloc[-i]) for i in range(2, 5)]
        prev_emas   = [float(hist[col].iloc[-i])     for i in range(2, 5)]
        came_from_above = all(c >= e for c, e in zip(prev_closes, prev_emas))

        if period == 20 and ema10_val:
            # EMA20: зона входа = от EMA20 до EMA10+1%.
            # Цена внутри этой зоны (пуллбэк или отскок) → Ready to Buy.
            # Выше EMA10+1% → сигнал снимается (сетап отыгран).
            if price_above and came_from_above:
                in_ema20_touch  = dist_pct <= entry_pct or low_wick_touch
                in_bounce_zone  = current_price <= ema10_val * 1.01
                if in_ema20_touch or in_bounce_zone:
                    signal_type = 'entry_zone'
            elif not price_above and came_from_above:
                if dist_pct <= entry_pct:
                    signal_type = 'entry_zone'
                elif dist_pct <= 2.0:
                    signal_type = 'watching'
        else:
            if price_above and came_from_above and dist_pct <= approach_pct:
                if dist_pct <= entry_pct or low_wick_touch:
                    signal_type = 'entry_zone'
                else:
                    # Approaching только если цена ПАДАЕТ к EMA (не отскакивает вверх)
                    lookback = 2 if is_weekly else 1
                    if len(hist) > lookback + 1:
                        prev_close = float(hist['Close'].iloc[-(lookback + 1)])
                        price_declining = current_price < prev_close
                    else:
                        price_declining = True
                    if price_declining:
                        signal_type = 'approaching'
            elif not price_above and came_from_above:
                lower_emas = [v for k, v in ema_values.items() if v < ema_val]
                touches_lower = any(current_price <= lv for lv in lower_emas)
                if not touches_lower:
                    if dist_pct <= entry_pct:
                        signal_type = 'entry_zone'
                    elif dist_pct <= approach_pct:
                        signal_type = 'watching'

        # EMA20: только в восходящем тренде — EMA20 и EMA50 обе должны расти
        if period == 20 and signal_type is not None and len(hist) > 20:
            ema20_past = float(hist[col].iloc[-21])
            ema50_now  = float(hist['ema_50'].iloc[-1])
            ema50_past = float(hist['ema_50'].iloc[-21])
            if not (ema_val > ema20_past and ema50_now > ema50_past):
                signal_type = None

        # EMA200: подавляем если 30 баров назад цена была ниже EMA200
        if period == 200 and signal_type is not None and len(hist) > 31:
            if float(hist['Close'].iloc[-31]) < float(hist[col].iloc[-31]):
                signal_type = None

        # EMA200: подавляем при медвежьем тренде (EMA50 < EMA200)
        if period == 200 and signal_type is not None:
            if float(hist['ema_50'].iloc[-1]) < ema_val:
                signal_type = None

        # EMA20: подавляем если цена упала слишком глубоко (уже у EMA50)
        if period == 20 and signal_type is not None:
            if current_price <= float(hist['ema_50'].iloc[-1]) * 1.01:
                signal_type = None

        # EMA50: снимаем если цена уже закрылась выше EMA20 (сетап отыгран)
        if period == 50 and signal_type is not None:
            ema20_val = ema_values.get('ema_20')
            if ema20_val and current_price > ema20_val:
                signal_type = None

        wick_touch = signal_type == 'entry_zone' and dist_pct > entry_pct and low_wick_touch

        result[col] = {
            'value': round(ema_val, 2),
            'atr': round(current_atr, 2),
            'distance_pct': dist_pct,
            'price_above': price_above,
            'signal_type': signal_type,
            'wick_touch': wick_touch,
        }

    # Одновременно не может гореть больше одной плашки — оставляем ближайшую EMA
    active = [(k, v) for k, v in result.items() if v.get('signal_type')]
    if len(active) > 1:
        closest_key = min(active, key=lambda x: x[1]['distance_pct'])[0]
        for k in result:
            if result[k].get('signal_type') and k != closest_key:
                result[k]['signal_type'] = None
                result[k]['wick_touch'] = False

    return result


def _is_tight_range(hist, lookback=15, max_range_pct=0.02):
    """True если последние lookback свечей торгуются в диапазоне < 2% (±1%).
    Такие акции в суперузкой консолидации — сигналы не актуальны."""
    if len(hist) < lookback:
        return False
    recent = hist.iloc[-lookback:]
    hi  = float(recent['High'].max())
    lo  = float(recent['Low'].min())
    mid = (hi + lo) / 2
    if mid <= 0:
        return False
    return (hi - lo) / mid < max_range_pct


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



            # Positive: Close закрылся выше EMA + 1 ATR (симметрично с негативом — оба по Close).
            # Если во время события LOW касался нижней EMA — позитив не считаем (неясно на что реакция).
            # Для weekly: требуем подтверждение — следующие 3 бара не должны закрыться ниже EMA−ATR.
            if f_close >= (f_ema + f_atr):
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
    Ищет молоты на недельном таймфрейме.
    Молот: нижняя тень >= 60%, тело <= 30%, верхняя тень <= 20% всей свечи.
    Зона: закрытие свечи между EMA200 и EMA20 (зона коррекции).
    Только завершённые недельные свечи: если сегодня пн-чт — текущая неделя исключается.
    """
    today_weekday = pd.Timestamp.now('UTC').weekday()  # 0=Mon … 4=Fri
    skip_last = 1 if today_weekday < 4 else 0          # пн-чт — свеча ещё не закрыта

    needed = lookback + skip_last + 1
    if len(hist_weekly) < needed:
        return None

    start   = -(lookback + skip_last)
    end     = -skip_last if skip_last > 0 else None
    candles = hist_weekly.iloc[start:end]
    if len(candles) == 0:
        return None

    hammers = []
    for i in range(len(candles)):
        try:
            o = float(candles['Open'].iloc[i])
            h = float(candles['High'].iloc[i])
            l = float(candles['Low'].iloc[i])
            c = float(candles['Close'].iloc[i])
        except Exception:
            continue

        candle_range = h - l
        if candle_range <= 0:
            continue

        body         = abs(c - o)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l

        body_pct         = body / candle_range
        upper_shadow_pct = upper_shadow / candle_range
        lower_shadow_pct = lower_shadow / candle_range

        if not (lower_shadow_pct >= 0.60 and body_pct <= 0.30 and upper_shadow_pct <= 0.20):
            continue

        # Зона коррекции: закрытие между EMA200 и EMA20
        try:
            ema20  = float(candles['ema_20'].iloc[i])
            ema200 = float(candles['ema_200'].iloc[i])
        except Exception:
            continue
        if not (ema200 <= c <= ema20):
            continue

        date_str = str(candles.index[i])[:10]
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
        # EMA10 — граница зоны EMA20 (не отдельный сигнал, только для EMA20 логики)
        hist_daily['ema_10'] = calculate_ema(hist_daily['Close'], 10)
        hist_daily = calculate_atr(hist_daily, 14)
        hist_daily = hist_daily.dropna()
        hist_daily.index = pd.to_datetime(hist_daily.index)
        cutoff_10y = hist_daily.index.max() - pd.DateOffset(years=10)
        hist_daily = hist_daily[hist_daily.index > cutoff_10y]

        # Weekly — то же самое
        for period in ema_periods:
            hist_weekly[f'ema_{period}'] = calculate_ema(hist_weekly['Close'], period)
        hist_weekly['ema_10']  = calculate_ema(hist_weekly['Close'], 10)
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

        # Суперузкий диапазон: акция в tight consolidation — все сигналы отключаем
        tight_range = _is_tight_range(hist_daily)
        if tight_range:
            print(f"  INFO: {ticker} in tight range (<4% over 15 bars) — skipping signals")

        # Плашки считаются отдельно для дневного и недельного таймфреймов
        # Если акция в боковике или в tight range — EMA сигналы не показываем
        if sideways_warning or tight_range:
            current_ema_daily = {}
            current_ema_weekly = {}
        else:
            current_ema_daily   = compute_current_ema_signals(hist_daily,   current_price, ema_periods, is_weekly=False)
            current_ema_weekly  = compute_current_ema_signals(hist_weekly,  current_price, ema_periods, is_weekly=True)

        # Паттерн "подхват скользящими": цена между EMA10 и EMA20 минимум 2 бара
        try:
            is_ready_20ema = False if tight_range else screener_module.screen_ready_20ema(hist_daily)
        except Exception:
            is_ready_20ema = False

        # Паттерн "двойной/тройной удар"
        try:
            if tight_range:
                strike_signal = None
            else:
                strike_daily  = screener_module.screen_double_triple_strike(hist_daily)
                strike_weekly = screener_module.screen_double_triple_strike(hist_weekly)
                strike_signals = []
                if strike_daily:
                    strike_daily['timeframe'] = 'daily'
                    strike_signals.append(strike_daily)
                if strike_weekly:
                    strike_weekly['timeframe'] = 'weekly'
                    strike_signals.append(strike_weekly)
                strike_signal = strike_signals if strike_signals else None
        except Exception:
            strike_signal = None

        result = {
            "ticker": ticker,
            "category": category,
            "current_price": round(current_price, 2),
            "downtrend_warning": downtrend_warning,
            "sideways_warning": sideways_warning,
            "ready_20ema": is_ready_20ema,
            "strike_signal": strike_signal,
            "current_ema": current_ema_daily,
            "current_ema_weekly": current_ema_weekly,
            # EMA10 хранится для fast_scan (верхняя граница зоны EMA20)
            "ema10_daily":    round(float(hist_daily['ema_10'].iloc[-1]),  2),
            "ema10_weekly":   round(float(hist_weekly['ema_10'].iloc[-1]), 2),
            "last_daily_date":  str(hist_daily.index[-1])[:10],
            "last_weekly_date": str(hist_weekly.index[-1])[:10],
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
                    min_ema_slope_pct=0.0 if period == 20 else 0.015,
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
                    min_ema_slope_bars=8 if period <= 50 else 0,
                    min_ema_slope_pct=0.0 if period == 20 else 0.015,
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

def _fetch_ticker(ticker, category):
    """Скачивает и анализирует один тикер — до 3 попыток при ошибке."""
    for attempt in range(3):
        signal = get_stock_signals(ticker, category)
        if signal is not None:
            return signal
        if attempt < 2:
            print(f"Retrying {ticker} in 10s... (attempt {attempt + 2}/3)")
            time.sleep(10)
    return None


def refresh_signals():
    """Обновить кэш сигналов — параллельно по 3 тикера."""
    global signals_cache
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ticker_list = load_tickers()
    signals = []
    signals_lock = threading.Lock()

    print(f"\n=== Starting analysis ({len(ticker_list)} tickers, sequential) ===")

    for i, (ticker, category) in enumerate(ticker_list):
        if i > 0:
            time.sleep(1)
        signal = _fetch_ticker(ticker, category)
        if signal:
            signals.append(signal)
    
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

def _sanitize(obj):
    """Рекурсивно заменяет NaN/Inf на None для совместимости с JSON."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.get("/api/signals")
def get_signals():
    """Возвращает сигналы из signals.json."""
    if should_refresh_cache():
        load_signals_from_file()

    with cache_lock:
        last_update = signals_cache['last_update']
        last_update_iso = last_update.isoformat() if last_update is not None else datetime.now().isoformat()
        data_generated_at = signals_cache.get('data_generated_at')
        return _sanitize({
            "signals": signals_cache['signals'],
            "last_update": last_update_iso,
            "data_generated_at": data_generated_at,
        })

@app.post("/api/refresh")
def refresh_signals_endpoint():
    """Запускает инкрементальное обновление EMA в фоне и сразу возвращает ответ."""
    thread = threading.Thread(target=incremental_update, daemon=True)
    thread.start()
    return {"status": "started"}

@app.get("/api/signals/{ticker}")
def get_signal_by_ticker(ticker: str):
    """Возвращает сигналы для конкретной акции"""
    signal = get_stock_signals(ticker.upper())
    if signal:
        return signal
    return {"error": f"Could not get data for {ticker}"}


@app.get("/api/screener")
def get_screener_results():
    """Возвращает последние результаты скринера."""
    path = screener_module.SCREENER_RESULTS_PATH
    if not path.exists():
        return {"last_update": None, "total_found": 0, "results": {}}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/touches/{ticker}")
def debug_touches(ticker: str, ema: int = 20, timeframe: str = "weekly",
                  years: int = 5, debug: bool = False):
    """
    Возвращает касания EMA. С ?debug=true показывает причину пропуска каждого бара.
    """
    try:
        yf = get_yfinance()
        hist = yf.download(ticker.upper(), period="15y",
                           interval="1wk" if timeframe == "weekly" else "1d",
                           progress=False, auto_adjust=False)
        hist = normalize_ohlc_columns(hist)
        if hist.empty:
            return {"error": f"No data for {ticker}"}

        for p in [10, 20, 50, 100, 200]:
            hist[f'ema_{p}'] = calculate_ema(hist['Close'], p)
        hist = calculate_atr(hist, 14)
        hist = hist.dropna()
        hist.index = pd.to_datetime(hist.index)

        cutoff = hist.index.max() - pd.DateOffset(years=years)
        hist = hist[hist.index > cutoff]

        ema_col   = f'ema_{ema}'
        all_cols  = ['ema_20', 'ema_50', 'ema_100', 'ema_200']
        lower_emas = [] if ema == 20 else [c for c in all_cols if c != ema_col]
        is_weekly  = timeframe == "weekly"
        near_pct   = 0.0015
        slope_bars = 8 if ema <= 50 else 0
        slope_pct  = 0.0 if ema == 20 else 0.015

        if is_weekly:
            touches = find_touch_events(
                hist, ema_col, 'atr',
                lower_ema_cols=lower_emas if lower_emas else None,
                cooldown_bars=2,
                require_rally_after_negative=True,
                min_ema_slope_bars=slope_bars,
                min_ema_slope_pct=slope_pct,
            )
        else:
            touches = find_touch_events(
                hist, ema_col, 'atr',
                lower_ema_cols=lower_emas if lower_emas else None,
                cooldown_bars=0,
                min_ema_slope_pct=slope_pct,
            )

        result = []
        for t in touches:
            result.append({
                "date":   str(t['date'])[:10],
                "result": t['result'],
                "price":  round(t['price'], 2),
                "ema":    round(t['ema'], 2),
                "atr":    round(t['atr'], 2),
            })

        positive = sum(1 for t in result if t['result'] == 'positive')
        negative = sum(1 for t in result if t['result'] == 'negative')

        response = {
            "ticker":    ticker.upper(),
            "ema":       ema_col,
            "timeframe": timeframe,
            "period":    f"last {years}y",
            "summary":   {"positive": positive, "negative": negative, "total": len(result)},
            "touches":   result,
        }

        if not debug:
            return response

        # Режим отладки: проверяем каждый бар и логируем причину пропуска
        skipped = []
        touch_dates = {t['date'] for t in result}
        last_neg_end = -10_000

        for i in range(1, len(hist) - 1):
            date_str = str(hist.index[i])[:10]
            if date_str in touch_dates:
                continue

            try:
                curr_close = float(hist['Close'].iloc[i])
                curr_low   = float(hist['Low'].iloc[i])
                ema_val    = float(hist[ema_col].iloc[i])
                atr_val    = float(hist['atr'].iloc[i])
            except Exception:
                continue

            if pd.isna(ema_val) or pd.isna(atr_val) or atr_val <= 0:
                continue

            near_from_above = ema_val <= curr_close <= ema_val * (1 + near_pct)
            actual_touch    = curr_low <= ema_val

            if not (near_from_above or actual_touch):
                continue  # не в зоне — не логируем, слишком много

            # Бар в зоне касания — выясняем почему пропущен
            reason = None

            came_from_above = all(
                float(hist['Close'].iloc[i - k]) > float(hist[ema_col].iloc[i - k])
                for k in range(1, 4) if i - k >= 0
            )
            if not came_from_above:
                reason = "came_from_above: один из 3 предыдущих баров ниже EMA"

            elif slope_bars > 0 and i >= slope_bars:
                ema_past = float(hist[ema_col].iloc[i - slope_bars])
                if ema_past > 0:
                    sp = (ema_val - ema_past) / ema_past
                    if sp < slope_pct:
                        reason = f"slope_filter: EMA выросла только на {sp*100:.2f}% за {slope_bars} баров (мин {slope_pct*100:.1f}%)"

            if reason:
                skipped.append({
                    "date":   date_str,
                    "price":  round(curr_close, 2),
                    "ema":    round(ema_val, 2),
                    "atr":    round(atr_val, 2),
                    "reason": reason,
                })

        response["skipped"] = skipped
        return response

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/screener/active")
def get_active_tickers():
    """Возвращает текущий активный список тикеров."""
    return screener_module.load_active_tickers()


@app.get("/api/analyze")
@app.post("/api/analyze")
def analyze_signals():
    """Агент: анализирует 10 случайных акций по EMA 20/50/200 daily+weekly, ищет аномалии."""
    import urllib.request, urllib.parse

    findings = analyzer_module.run_analysis(
        signals_path=SIGNALS_JSON_PATH,
        find_touch_fn=find_touch_events,
        calc_ema_fn=calculate_ema,
        calc_atr_fn=calculate_atr,
        normalize_fn=normalize_ohlc_columns,
        get_yf_fn=get_yfinance,
    )

    if not findings:
        return {"status": "ok", "detail": "Аномалий не найдено"}

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if bot_token and chat_id:
        now_str = pd.Timestamp.now("Europe/Moscow").strftime("%b %d, %Y")
        text = f"🔍 Анализ логики — {now_str}\n\n" + "\n\n".join(findings)
        if len(text) > 4096:
            text = text[:4090] + "..."
        url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                json.loads(resp.read())
        except Exception:
            pass

    return {"status": "ok", "findings": findings}


@app.get("/api/telegram/debug")
def telegram_debug():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    return {
        "token_set": bool(token),
        "token_len": len(token),
        "chat_set":  bool(chat),
        "chat_id":   chat,
    }


@app.get("/api/telegram/send")
def telegram_send_get():
    return _telegram_send_impl()


@app.get("/api/telegram/report")
@app.post("/api/telegram/report")
def telegram_report():
    """Фильтрованный отчёт: entry_zone, prob>=65%, восходящий тренд, EMA 20/50/200 + Ready 20 EMA."""
    import urllib.request, urllib.parse

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return {"status": "error", "detail": "TELEGRAM_BOT_TOKEN или CHAT_ID не заданы"}

    try:
        with open(SIGNALS_JSON_PATH) as f:
            data = json.load(f)
        signals = data.get("signals", data) if isinstance(data, dict) else data
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    tickers = []
    for s in signals:
        ticker = s.get("ticker", "")

        ema_d = s.get("current_ema") or {}

        # Фильтр восходящего тренда: цена должна быть выше дневной EMA200
        ema200_daily = ema_d.get("ema_200", {})
        if not (isinstance(ema200_daily, dict) and ema200_daily.get("price_above", False)):
            continue

        # Ready 20 EMA — без фильтра вероятности
        if s.get("ready_20ema"):
            tickers.append(ticker)
            continue

        ema_w = s.get("current_ema_weekly") or {}
        p1_d  = (s.get("daily")   or {}).get("period_1_5y") or {}
        p1_w  = (s.get("weekly")  or {}).get("period_1_5y") or {}

        found = False
        for ema_key in ["ema_20", "ema_50", "ema_200"]:
            # Daily
            v = ema_d.get(ema_key, {})
            if isinstance(v, dict) and v.get("signal_type") == "entry_zone":
                prob = (p1_d.get(ema_key) or {}).get("probability", 0)
                if prob >= 65:
                    found = True
                    break
            # Weekly
            v = ema_w.get(ema_key, {})
            if isinstance(v, dict) and v.get("signal_type") == "entry_zone":
                prob = (p1_w.get(ema_key) or {}).get("probability", 0)
                if prob >= 65:
                    found = True
                    break

        if found:
            tickers.append(ticker)

    if not tickers:
        return {"status": "ok", "detail": "Нет сигналов по критериям"}

    now_str = pd.Timestamp.now("Europe/Moscow").strftime("%b %d, %Y")
    text = f"{now_str} filtered\n{', '.join(tickers)}"

    url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()

    try:
        req = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                return {"status": "ok", "tickers": tickers}
            else:
                return {"status": "error", "detail": result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/telegram/send")
def telegram_send_endpoint():
    """Отправляет сигналы в Telegram и возвращает подробный результат."""
    try:
        return _telegram_send_impl()
    except Exception as e:
        return {"status": "error", "detail": f"Unexpected: {e}"}

def _telegram_send_impl():
    import urllib.request, urllib.parse

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token:
        return {"status": "error", "detail": "TELEGRAM_BOT_TOKEN не задан"}
    if not chat_id:
        return {"status": "error", "detail": "TELEGRAM_CHAT_ID не задан"}

    try:
        with open(SIGNALS_JSON_PATH) as f:
            data = json.load(f)
        signals = data.get("signals", data) if isinstance(data, dict) else data
    except Exception as e:
        return {"status": "error", "detail": f"signals.json: {e}"}

    tickers_with_signals = []
    for s in signals:
        ticker = s.get("ticker", "")
        ema_d = s.get("current_ema") or {}
        ema_w = s.get("current_ema_weekly") or {}
        has_ema = any(
            isinstance(v, dict) and v.get("signal_type") in ("entry_zone", "approaching", "watching")
            for d in (ema_d, ema_w) for k, v in d.items() if k != "ema_100"
        )
        if has_ema or s.get("ready_20ema") or s.get("strike_signal") or s.get("hammer_signal"):
            tickers_with_signals.append(ticker)

    if not tickers_with_signals:
        return {"status": "ok", "detail": "Нет акций с сигналами"}

    now_str = pd.Timestamp.now("America/New_York").strftime("%b %d, %Y")
    text = f"{now_str}\n{', '.join(tickers_with_signals)}"

    url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id":    chat_id,
        "text":       text,
    }).encode()

    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                return {"status": "ok", "tickers": tickers_with_signals}
            else:
                return {"status": "error", "detail": result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/screener/run")
def run_screener_endpoint():
    """Запускает полный скан вселенной в фоновом потоке (~10-15 мин)."""
    def _run():
        yf = get_yfinance()
        screener_module.run_full_screener(yf)
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "message": "Full screener started (~10-15 min)"}


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

def _is_data_stale(max_age_hours=12) -> bool:
    """Возвращает True если данные устарели или отсутствуют."""
    with cache_lock:
        generated_at = signals_cache.get('data_generated_at')
    if not generated_at:
        return True
    try:
        generated_ts = pd.Timestamp(generated_at)
        if generated_ts.tzinfo is None:
            generated_ts = generated_ts.tz_localize('UTC')
        age_hours = (pd.Timestamp.now('UTC') - generated_ts).total_seconds() / 3600
        return age_hours > max_age_hours
    except Exception:
        return True


def compute_fast_ema_signals(current_price, current_low, stored_ema,
                             is_weekly=False, prev_price=None, ema10=None):
    """
    Быстрая проверка сигналов по текущей цене без загрузки истории.
    Использует сохранённые EMA уровни из последнего полного пересчёта.
    """
    entry_pct    = 2.0 if is_weekly else 1.0
    wick_pct     = 2.0 if is_weekly else 1.0
    approach_pct = 4.0 if is_weekly else 2.0

    # Глобальный фильтр тренда: цена ниже 200 EMA = нисходящий тренд, сигналы не даём
    _ema200 = stored_ema.get('ema_200', {})
    _ema200_val = float(_ema200.get('value', 0)) if _ema200 else 0
    if _ema200_val and current_price < _ema200_val:
        result = {}
        for ema_key, stored in stored_ema.items():
            ema_val = stored.get('value')
            dist_pct = round(abs(current_price - ema_val) / ema_val * 100, 2) if ema_val else 0
            result[ema_key] = {
                'value': ema_val,
                'atr': stored.get('atr'),
                'distance_pct': dist_pct,
                'price_above': (current_price >= ema_val) if ema_val else False,
                'signal_type': None,
                'wick_touch': False,
            }
        return result

    result = {}
    for ema_key, stored in stored_ema.items():
        ema_val = stored.get('value')
        if not ema_val:
            result[ema_key] = stored
            continue

        try:
            period = int(ema_key.replace('ema_', ''))
        except ValueError:
            result[ema_key] = stored
            continue

        dist_pct = round(abs(current_price - ema_val) / ema_val * 100, 2)
        price_above = current_price >= ema_val
        low_wick_touch = price_above and current_low <= ema_val * (1 + wick_pct / 100)
        was_above = stored.get('price_above', True)
        signal_type = None

        if period == 20 and ema10 is not None:
            # EMA20: зона входа = EMA20 → EMA10+1%
            if was_above and price_above:
                in_ema20_touch = dist_pct <= entry_pct or low_wick_touch
                in_bounce_zone = current_price <= ema10 * 1.01
                if in_ema20_touch or in_bounce_zone:
                    signal_type = 'entry_zone'
            elif was_above and not price_above:
                if dist_pct <= entry_pct:
                    signal_type = 'entry_zone'
                elif dist_pct <= 2.0:
                    signal_type = 'watching'
        else:
            if was_above:
                if price_above and dist_pct <= approach_pct:
                    if dist_pct <= entry_pct or low_wick_touch:
                        signal_type = 'entry_zone'
                    else:
                        # Approaching только когда цена падает к EMA (не растёт)
                        price_declining = prev_price is None or current_price <= prev_price
                        if price_declining:
                            signal_type = 'approaching'
                elif not price_above and dist_pct <= approach_pct:
                    if dist_pct <= entry_pct:
                        signal_type = 'entry_zone'
                    else:
                        signal_type = 'watching'

        # EMA20: подавляем если цена упала слишком глубоко (уже у EMA50)
        if period == 20 and signal_type is not None:
            _ema50 = stored_ema.get('ema_50', {})
            _ema50_val = float(_ema50.get('value', 0)) if _ema50 else 0
            if _ema50_val and current_price <= _ema50_val * 1.01:
                signal_type = None

        # EMA50: снимаем если цена уже закрылась выше EMA20 (сетап отыгран)
        if period == 50 and signal_type is not None:
            _ema20 = stored_ema.get('ema_20', {})
            _ema20_val = float(_ema20.get('value', 0)) if _ema20 else 0
            if _ema20_val and current_price > _ema20_val:
                signal_type = None

        result[ema_key] = {
            'value': ema_val,
            'atr': stored.get('atr'),
            'distance_pct': dist_pct,
            'price_above': price_above,
            'signal_type': signal_type,
            'wick_touch': signal_type == 'entry_zone' and dist_pct > entry_pct and low_wick_touch,
        }

    # Одновременно не может гореть больше одной плашки — оставляем ближайшую EMA
    active = [(k, v) for k, v in result.items() if v.get('signal_type')]
    if len(active) > 1:
        closest_key = min(active, key=lambda x: x[1]['distance_pct'])[0]
        for k in result:
            if result[k].get('signal_type') and k != closest_key:
                result[k]['signal_type'] = None
                result[k]['wick_touch'] = False

    return result


def fast_scan_signals():
    """
    Быстрое обновление сигналов — батч-загрузка текущих цен всех тикеров одним запросом.
    Сравнивает с сохранёнными EMA уровнями из последнего полного пересчёта.
    Выполняется за 45 минут до закрытия NYSE (~30 секунд для 470 тикеров).
    """
    with cache_lock:
        existing = {s['ticker']: s for s in signals_cache.get('signals', [])}

    if not existing:
        print("[FAST-SCAN] No cached signals — running full refresh instead")
        refresh_signals()
        return

    tickers = list(existing.keys())
    print(f"[FAST-SCAN] Batch downloading prices for {len(tickers)} tickers...")

    yf = get_yfinance()
    try:
        batch = yf.download(
            tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(f"[FAST-SCAN] Batch download failed: {e}")
        return

    updated = []
    for ticker, signal in existing.items():
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                ticker_data = batch.xs(ticker, level=1, axis=1).dropna()
            else:
                ticker_data = batch.dropna()

            if ticker_data.empty:
                updated.append(signal)
                continue

            cur_price  = float(ticker_data['Close'].iloc[-1])
            cur_low    = float(ticker_data['Low'].iloc[-1])
            prev_price = float(ticker_data['Close'].iloc[-2]) if len(ticker_data) >= 2 else None

            ema10_daily   = signal.get('ema10_daily')
            ema10_weekly  = signal.get('ema10_weekly')

            new_signal = dict(signal)
            new_signal['current_price'] = round(cur_price, 2)

            if signal.get('current_ema'):
                new_signal['current_ema'] = compute_fast_ema_signals(
                    cur_price, cur_low, signal['current_ema'],
                    is_weekly=False, prev_price=prev_price, ema10=ema10_daily
                )
            if signal.get('current_ema_weekly'):
                new_signal['current_ema_weekly'] = compute_fast_ema_signals(
                    cur_price, cur_low, signal['current_ema_weekly'],
                    is_weekly=True, prev_price=prev_price, ema10=ema10_weekly
                )

            updated.append(new_signal)
        except Exception as e:
            print(f"[FAST-SCAN] Error for {ticker}: {e}")
            updated.append(signal)

    now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    with cache_lock:
        signals_cache['signals'] = updated
        signals_cache['last_update'] = pd.Timestamp.now()
        signals_cache['data_generated_at'] = now_str

    try:
        SIGNALS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({'last_update': now_str, 'signals': updated}, f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[FAST-SCAN] Failed to save: {e}")

    print(f"[FAST-SCAN] Done! {len(updated)} signals updated")


def _apply_ema_incremental(new_bars, stored_ema, stored_ema10, ema_periods, last_date, is_weekly=False):
    """
    Обновляет EMA по формуле EMA(t) = α*Close + (1-α)*EMA(t-1) только для новых баров.
    Возвращает (updated_stored_ema, updated_ema10) или (None, stored_ema10) при ошибке.
    """
    if new_bars.empty or not stored_ema:
        return None, stored_ema10

    new_bars = new_bars.copy()
    new_bars.index = pd.to_datetime(new_bars.index)

    # Только бары новее last_date
    if last_date:
        try:
            last_ts = pd.Timestamp(last_date).date()
            bars_to_apply = new_bars[new_bars.index.date > last_ts]
        except Exception:
            bars_to_apply = new_bars.iloc[-2:]
    else:
        bars_to_apply = new_bars  # первый запуск — берём все доступные

    # Извлекаем сохранённые EMA значения
    ema_vals = {}
    for ema_key, data in stored_ema.items():
        try:
            period = int(ema_key.replace('ema_', ''))
            val = data.get('value')
            if val:
                ema_vals[period] = float(val)
        except (ValueError, TypeError):
            pass

    ema10_val = float(stored_ema10) if stored_ema10 else None

    # Применяем формулу для каждого нового бара
    if not bars_to_apply.empty:
        for i in range(len(bars_to_apply)):
            try:
                close = float(bars_to_apply['Close'].iloc[i])
            except Exception:
                continue
            for period in ema_periods:
                if period in ema_vals:
                    alpha = 2.0 / (period + 1)
                    ema_vals[period] = alpha * close + (1 - alpha) * ema_vals[period]
            if ema10_val is not None:
                ema10_val = (2.0 / 11) * close + (9.0 / 11) * ema10_val

    current_price = float(new_bars['Close'].iloc[-1])
    current_low   = float(new_bars['Low'].iloc[-1])
    prev_price    = float(new_bars['Close'].iloc[-2]) if len(new_bars) >= 2 else None

    # Сохраняем старый price_above как was_above для следующего цикла
    ema_for_signal = {}
    for ema_key, data in stored_ema.items():
        try:
            period = int(ema_key.replace('ema_', ''))
        except ValueError:
            ema_for_signal[ema_key] = data
            continue
        new_val = ema_vals.get(period)
        if new_val is None:
            ema_for_signal[ema_key] = data
            continue
        ema_for_signal[ema_key] = {
            'value':        round(new_val, 4),
            'atr':          data.get('atr'),
            'price_above':  data.get('price_above', True),  # was_above для следующего вызова
            'distance_pct': round(abs(current_price - new_val) / new_val * 100, 2),
            'signal_type':  data.get('signal_type'),
            'wick_touch':   data.get('wick_touch', False),
        }

    result = compute_fast_ema_signals(
        current_price, current_low, ema_for_signal,
        is_weekly=is_weekly, prev_price=prev_price, ema10=ema10_val,
    )
    return result, (round(ema10_val, 4) if ema10_val else None)


def incremental_update():
    """
    Инкрементальное обновление: скачиваем только последние 10 дней батчем (~5 сек),
    обновляем EMA по формуле. Запускается ежедневно в 16:30 ET (Mon-Fri).
    Полный пересчёт (refresh_signals) — только по воскресеньям для калибровки.
    """
    import time as _time
    start = _time.time()

    with cache_lock:
        existing = {s['ticker']: s for s in signals_cache.get('signals', [])}

    if not existing:
        print("[INCREMENTAL] No cached signals — running full refresh")
        refresh_signals()
        return

    all_tickers    = load_tickers()
    new_tickers    = [(t, c) for t, c in all_tickers if t not in existing]
    tickers        = list(existing.keys())
    ema_periods    = [20, 50, 100, 200]
    yf             = get_yfinance()

    print(f"[INCREMENTAL] Batch downloading {len(tickers)} tickers (10d daily + 1mo weekly)...")
    try:
        batch_daily  = yf.download(tickers, period="10d", interval="1d",  progress=False, auto_adjust=False)
        batch_weekly = yf.download(tickers, period="1mo", interval="1wk", progress=False, auto_adjust=False)
    except Exception as e:
        print(f"[INCREMENTAL] Download failed: {e}")
        return

    updated = []
    for ticker, signal in existing.items():
        try:
            if isinstance(batch_daily.columns, pd.MultiIndex):
                daily_new  = batch_daily.xs(ticker,  level=1, axis=1).dropna()
                weekly_new = batch_weekly.xs(ticker, level=1, axis=1).dropna()
            else:
                daily_new  = batch_daily.dropna()
                weekly_new = batch_weekly.dropna()

            if daily_new.empty:
                updated.append(signal)
                continue

            new_signal = dict(signal)
            new_signal['current_price'] = round(float(daily_new['Close'].iloc[-1]), 2)

            if signal.get('current_ema'):
                new_ema_d, new_ema10_d = _apply_ema_incremental(
                    daily_new, signal['current_ema'],
                    signal.get('ema10_daily'), ema_periods,
                    signal.get('last_daily_date'), is_weekly=False,
                )
                if new_ema_d:
                    new_signal['current_ema'] = new_ema_d
                    if new_ema10_d is not None:
                        new_signal['ema10_daily'] = new_ema10_d
                    new_signal['last_daily_date'] = str(daily_new.index[-1])[:10]

            if signal.get('current_ema_weekly') and not weekly_new.empty:
                new_ema_w, new_ema10_w = _apply_ema_incremental(
                    weekly_new, signal['current_ema_weekly'],
                    signal.get('ema10_weekly'), ema_periods,
                    signal.get('last_weekly_date'), is_weekly=True,
                )
                if new_ema_w:
                    new_signal['current_ema_weekly'] = new_ema_w
                    if new_ema10_w is not None:
                        new_signal['ema10_weekly'] = new_ema10_w
                    new_signal['last_weekly_date'] = str(weekly_new.index[-1])[:10]

            updated.append(new_signal)
        except Exception as e:
            print(f"[INCREMENTAL] Error for {ticker}: {e}")
            updated.append(signal)

    if new_tickers:
        print(f"[INCREMENTAL] Adding {len(new_tickers)} new tickers (full calc)...")
        for t, cat in new_tickers:
            time.sleep(0.5)
            sig = _fetch_ticker(t, cat)
            if sig:
                updated.append(sig)

    now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    with cache_lock:
        signals_cache['signals'] = updated
        signals_cache['last_update'] = pd.Timestamp.now()
        signals_cache['data_generated_at'] = now_str
    try:
        SIGNALS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({'last_update': now_str, 'signals': updated}, f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[INCREMENTAL] Failed to save: {e}")

    print(f"[INCREMENTAL] Done! {len(updated)} signals in {_time.time() - start:.1f}s")


def send_telegram_report():
    """Отправляет все текущие сигналы с плашками в Telegram (14:00 ET по будням)."""
    import urllib.request
    import urllib.parse

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        print("[TELEGRAM] BOT_TOKEN или CHAT_ID не заданы — пропускаем")
        return

    try:
        with open(SIGNALS_JSON_PATH) as f:
            data = json.load(f)
        signals = data.get("signals", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[TELEGRAM] Не удалось прочитать signals.json: {e}")
        return

    tickers_with_signals = []
    for s in signals:
        ticker = s.get("ticker", "")
        ema_d = s.get("current_ema") or {}
        ema_w = s.get("current_ema_weekly") or {}
        has_ema = any(
            isinstance(v, dict) and v.get("signal_type") in ("entry_zone", "approaching", "watching")
            for d in (ema_d, ema_w) for k, v in d.items() if k != "ema_100"
        )
        if has_ema or s.get("ready_20ema") or s.get("strike_signal") or s.get("hammer_signal"):
            tickers_with_signals.append(ticker)

    if not tickers_with_signals:
        print("[TELEGRAM] Нет сигналов для отправки")
        return

    now_str = pd.Timestamp.now("America/New_York").strftime("%b %d, %Y")
    text = f"{now_str}\n{', '.join(tickers_with_signals)}"

    if len(text) > 4096:
        text = text[:4090] + "..."

    url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }).encode()

    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                print(f"[TELEGRAM] Отправлено {total} акций")
            else:
                print(f"[TELEGRAM] Ошибка API: {result}")
    except Exception as e:
        print(f"[TELEGRAM] Ошибка отправки: {e}")


def auto_refresh_background():
    """
    Расписание (ET):
      Пн-Пт 14:00 — send_telegram_report(): дайджест сигналов
      Пн-Пт 15:15 — fast_scan_signals()   : текущие цены, ~30 сек
      Пн-Пт 16:30 — incremental_update()  : инкрементальный пересчёт EMA, ~1 мин
      Воскресенье 10:00 — refresh_signals(): полный пересчёт раз в неделю
    """
    import zoneinfo
    global background_thread_stop

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    print("[AUTO-REFRESH] Started. Schedule: fast 15:15 + incremental 16:30 (Mon-Fri), screener 23:00 (daily), full rebuild Sun 10:00 ET")

    while not background_thread_stop:
        try:
            now_et   = pd.Timestamp.now(et_tz)
            today_et = now_et.normalize()

            dow = now_et.dayofweek  # 0=Mon … 6=Sun

            candidates = []

            if dow < 5:  # Пн-Пт
                fast_scan_et   = today_et + pd.Timedelta(hours=15, minutes=15)
                incremental_et = today_et + pd.Timedelta(hours=16, minutes=30)
                screener_et    = today_et + pd.Timedelta(hours=23, minutes=0)
                if now_et < fast_scan_et:
                    candidates.append(('fast_scan',   fast_scan_et))
                if now_et < incremental_et:
                    candidates.append(('incremental', incremental_et))
                if now_et < screener_et:
                    candidates.append(('screener',    screener_et))

            elif dow == 5:  # Суббота
                screener_et = today_et + pd.Timedelta(hours=23, minutes=0)
                if now_et < screener_et:
                    candidates.append(('screener', screener_et))

            elif dow == 6:  # Воскресенье
                screener_et   = today_et + pd.Timedelta(hours=8,  minutes=0)
                full_build_et = today_et + pd.Timedelta(hours=10, minutes=0)
                if now_et < screener_et:
                    candidates.append(('screener',    screener_et))
                if now_et < full_build_et:
                    candidates.append(('full_build', full_build_et))

            if not candidates:
                # Ищем следующее ближайшее событие
                check = now_et + pd.Timedelta(days=1)
                for _ in range(7):
                    check_day = check.normalize()
                    if check.dayofweek < 5:
                        candidates.append(('fast_scan', check_day + pd.Timedelta(hours=15, minutes=15)))
                        break
                    elif check.dayofweek == 6:
                        candidates.append(('full_build', check_day + pd.Timedelta(hours=10, minutes=0)))
                        break
                    check += pd.Timedelta(days=1)

            if not candidates:
                time.sleep(3600)
                continue

            kind, run_at = candidates[0]
            wait_seconds = (run_at - now_et).total_seconds()
            if wait_seconds < 0:
                wait_seconds = 0

            print(f"[AUTO-REFRESH] Next: {kind} at {run_at.strftime('%Y-%m-%d %H:%M ET')} "
                  f"(in {int(wait_seconds/3600)}h {int((wait_seconds % 3600)/60)}m)")

            time.sleep(max(wait_seconds, 1))

            if background_thread_stop:
                break

            if kind == 'fast_scan':
                print(f"[AUTO-REFRESH] Starting fast scan at {pd.Timestamp.now()}")
                fast_scan_signals()
                print(f"[AUTO-REFRESH] Fast scan completed at {pd.Timestamp.now()}")
            elif kind == 'incremental':
                print(f"[AUTO-REFRESH] Starting incremental update at {pd.Timestamp.now()}")
                incremental_update()
                print(f"[AUTO-REFRESH] Incremental update completed at {pd.Timestamp.now()}")
            elif kind == 'screener':
                print(f"[AUTO-REFRESH] Starting screener at {pd.Timestamp.now()}")
                screener_module.run_full_screener(get_yfinance())
                print(f"[AUTO-REFRESH] Screener completed at {pd.Timestamp.now()}")
            else:
                print(f"[AUTO-REFRESH] Starting full rebuild at {pd.Timestamp.now()}")
                refresh_signals()
                print(f"[AUTO-REFRESH] Full rebuild completed at {pd.Timestamp.now()}")

        except Exception as e:
            print(f"[AUTO-REFRESH] Error: {e}")
            time.sleep(60)


@app.on_event("startup")
async def startup_event():
    """Загружаем сигналы при старте и запускаем фоновое расписание."""
    loaded = load_signals_from_file()
    if loaded:
        print("\n=== Signals loaded from file on startup ===")
        if ENABLE_BACKGROUND_REFRESH:
            if _is_data_stale(max_age_hours=168):  # >7 дней — полный пересчёт
                print("=== Data is stale (>7d) — triggering full refresh ===")
                t = threading.Thread(target=refresh_signals, daemon=True)
                t.start()
            elif _is_data_stale(max_age_hours=12):  # 12ч–7д — инкрементальное обновление
                print("=== Data is stale (>12h) — triggering incremental update ===")
                t = threading.Thread(target=incremental_update, daemon=True)
                t.start()
    elif ENABLE_STARTUP_REFRESH:
        try:
            print("\n=== Loading signals on startup via yfinance ===")
            refresh_signals()
        except Exception as e:
            print(f"\n⚠️  Startup refresh failed: {e}")
    else:
        print("\n=== No signals file found, triggering background refresh ===")
        if not os.getenv("VERCEL") and ENABLE_BACKGROUND_REFRESH:
            t = threading.Thread(target=refresh_signals, daemon=True)
            t.start()

    # Запускаем фоновое расписание — работает на Railway и локально
    if ENABLE_BACKGROUND_REFRESH and not os.getenv("VERCEL"):
        bg = threading.Thread(target=auto_refresh_background, daemon=True)
        bg.start()
        print("=== Auto-refresh scheduler started (fast 15:15 + incremental 16:30 ET Mon-Fri, full rebuild Sun 10:00 ET) ===")


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*50)
    print("StockScanner Backend Starting...")
    print("Auto-refresh: fast 15:15 + incremental 16:30 ET (Mon-Fri), full rebuild Sun 10:00 ET")
    print(f"API: http://127.0.0.1:{SERVER_PORT}")
    print("="*50 + "\n")

    try:
        uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
    except OSError as e:
        if "10048" in str(e) or "Address already in use" in str(e):
            fallback_port = SERVER_PORT + 1
            print(f"Port {SERVER_PORT} already in use. Trying port {fallback_port}...")
            uvicorn.run(app, host="127.0.0.1", port=fallback_port)
        else:
            raise