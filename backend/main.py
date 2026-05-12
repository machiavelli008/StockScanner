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
    # При параллельных загрузках yfinance может вернуть дублирующиеся колонки
    df = df.loc[:, ~df.columns.duplicated()]
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
    current_low = float(hist['Low'].iloc[-1])
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
        # Low свечи коснулся зоны EMA: тень дошла до уровня (0.15% выше или ниже EMA)
        low_wick_touch = price_above and current_low <= ema_val * 1.0015
        signal_type = None

        if price_above:
            # Проверяем что последние 3 свечи тоже были выше EMA (цена снижается сверху вниз)
            prev_closes = [float(hist['Close'].iloc[-i]) for i in range(2, 5)]
            prev_emas = [float(hist[col].iloc[-i]) for i in range(2, 5)]
            came_from_above = all(c >= e for c, e in zip(prev_closes, prev_emas))
            if came_from_above:
                if dist_pct <= 0.15 or low_wick_touch:
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

                if came_from_above:
                    if dist_pct <= 0.15:
                        signal_type = 'entry_zone'  # только что упала — касание
                    elif dist_pct <= 2.0:
                        signal_type = 'watching'    # только что упала ниже EMA

        # EMA20: только в восходящем тренде — EMA20 и EMA50 обе должны расти
        if period == 20 and signal_type is not None and len(hist) > 20:
            ema20_past = float(hist[col].iloc[-21])
            ema50_now = float(hist['ema_50'].iloc[-1])
            ema50_past = float(hist['ema_50'].iloc[-21])
            ema20_rising = ema_val > ema20_past
            ema50_rising = ema50_now > ema50_past
            if not (ema20_rising and ema50_rising):
                signal_type = None

        # EMA200: подавляем если 30 баров назад цена была ниже EMA200 (пришла снизу вверх)
        if period == 200 and signal_type is not None and len(hist) > 31:
            close_30_ago = float(hist['Close'].iloc[-31])
            ema200_30_ago = float(hist[col].iloc[-31])
            if close_30_ago < ema200_30_ago:
                signal_type = None

        # EMA200: подавляем если EMA50 ниже EMA200 (медвежий тренд — нет золотого креста)
        if period == 200 and signal_type is not None:
            ema50_val = float(hist['ema_50'].iloc[-1])
            if ema50_val < ema_val:
                signal_type = None

        # EMA20: подавляем если текущая цена уже у EMA50 (цена упала слишком глубоко)
        if period == 20 and signal_type is not None:
            ema50_val = float(hist['ema_50'].iloc[-1])
            if current_price <= ema50_val * 1.01:
                signal_type = None

        # wick_touch=True только когда сигнал пришёл именно от тени (Close был дальше 0.15%)
        wick_touch = signal_type == 'entry_zone' and dist_pct > 0.15 and low_wick_touch

        result[col] = {
            'value': round(ema_val, 2),
            'distance_pct': dist_pct,
            'price_above': price_above,
            'signal_type': signal_type,
            'wick_touch': wick_touch,
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
        # Если акция в боковике — EMA сигналы не показываем
        if sideways_warning:
            current_ema_daily = {}
            current_ema_weekly = {}
        else:
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

    print(f"\n=== Starting analysis ({len(ticker_list)} tickers, 3 workers) ===")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_ticker, ticker, category): ticker
            for ticker, category in ticker_list
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                signal = future.result()
                if signal:
                    with signals_lock:
                        signals.append(signal)
            except Exception as e:
                print(f"Error for {ticker}: {e}")
    
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


def compute_fast_ema_signals(current_price, current_low, stored_ema):
    """
    Быстрая проверка сигналов по текущей цене без загрузки истории.
    Использует сохранённые EMA уровни из последнего полного пересчёта.
    """
    result = {}
    for ema_key, stored in stored_ema.items():
        ema_val = stored.get('value')
        if not ema_val:
            result[ema_key] = stored
            continue

        dist_pct = round(abs(current_price - ema_val) / ema_val * 100, 2)
        price_above = current_price >= ema_val
        low_wick_touch = price_above and current_low <= ema_val * 1.0015

        # Используем сохранённый price_above как признак что цена пришла сверху
        was_above = stored.get('price_above', True)
        signal_type = None

        if was_above:
            if price_above:
                if dist_pct <= 0.15 or low_wick_touch:
                    signal_type = 'entry_zone'
                elif dist_pct <= 1.0:
                    signal_type = 'approaching'
            else:
                if dist_pct <= 0.15:
                    signal_type = 'entry_zone'
                elif dist_pct <= 2.0:
                    signal_type = 'watching'

        result[ema_key] = {
            'value': ema_val,
            'distance_pct': dist_pct,
            'price_above': price_above,
            'signal_type': signal_type,
            'wick_touch': signal_type == 'entry_zone' and dist_pct > 0.15 and low_wick_touch,
        }
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

            cur_price = float(ticker_data['Close'].iloc[-1])
            cur_low   = float(ticker_data['Low'].iloc[-1])

            new_signal = dict(signal)
            new_signal['current_price'] = round(cur_price, 2)

            if signal.get('current_ema'):
                new_signal['current_ema'] = compute_fast_ema_signals(
                    cur_price, cur_low, signal['current_ema']
                )
            if signal.get('current_ema_weekly'):
                new_signal['current_ema_weekly'] = compute_fast_ema_signals(
                    cur_price, cur_low, signal['current_ema_weekly']
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


def auto_refresh_background():
    """
    Расписание (ET, торговые дни Пн-Пт):
      15:15 ET — fast_scan_signals()  : текущие цены, ~30 сек, клиент видит сигналы
      16:30 ET — refresh_signals()    : полный пересчёт с историей после закрытия свечи
    """
    import zoneinfo
    global background_thread_stop

    et_tz = zoneinfo.ZoneInfo("America/New_York")
    print("[AUTO-REFRESH] Started. Schedule: fast scan 15:15 ET, full rebuild 16:30 ET (Mon-Fri)")

    while not background_thread_stop:
        try:
            now_et = pd.Timestamp.now(et_tz)
            today  = now_et.normalize().tz_localize(None)
            today_et = pd.Timestamp(today).tz_localize(et_tz)

            # Расписание на сегодня
            fast_scan_et  = today_et + pd.Timedelta(hours=15, minutes=15)
            full_build_et = today_et + pd.Timedelta(hours=16, minutes=30)

            # Выбираем ближайшее будущее событие
            candidates = []
            if now_et < fast_scan_et:
                candidates.append(('fast_scan',  fast_scan_et))
            if now_et < full_build_et:
                candidates.append(('full_build', full_build_et))

            if not candidates:
                # Оба события сегодня прошли — переходим к следующему торговому дню
                next_day = today_et + pd.Timedelta(days=1)
                while next_day.dayofweek >= 5:
                    next_day += pd.Timedelta(days=1)
                candidates.append(('fast_scan', next_day + pd.Timedelta(hours=15, minutes=15)))

            kind, run_at = candidates[0]

            # Пропускаем выходные
            while run_at.dayofweek >= 5:
                run_at += pd.Timedelta(days=1)

            wait_seconds = (run_at - now_et).total_seconds()
            print(f"[AUTO-REFRESH] Next: {kind} at {run_at.strftime('%Y-%m-%d %H:%M ET')} "
                  f"(in {int(wait_seconds/3600)}h {int((wait_seconds % 3600)/60)}m)")

            time.sleep(wait_seconds)

            if background_thread_stop:
                break

            if kind == 'fast_scan':
                print(f"[AUTO-REFRESH] Starting fast scan at {pd.Timestamp.now()}")
                fast_scan_signals()
                print(f"[AUTO-REFRESH] Fast scan completed at {pd.Timestamp.now()}")
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
        if _is_data_stale(max_age_hours=12) and ENABLE_BACKGROUND_REFRESH:
            print("=== Data is stale (>12h) — triggering background refresh ===")
            t = threading.Thread(target=refresh_signals, daemon=True)
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
        print("=== Auto-refresh scheduler started (15:15 fast scan + 16:30 full rebuild ET) ===")


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*50)
    print("StockScanner Backend Starting...")
    print("Auto-refresh: fast scan 15:15 ET + full rebuild 16:30 ET (Mon-Fri)")
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