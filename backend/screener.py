"""
Screener module — ежедневный скан всех акций США.
Находит акции по критериям, поддерживает динамический список тикеров.
"""
import json
import time
import requests
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR              = Path(__file__).parent.parent / "data"
SCREENER_RESULTS_PATH = DATA_DIR / "screener_results.json"
ACTIVE_TICKERS_PATH   = DATA_DIR / "active_tickers.json"

MIN_PRICE            = 5.0
MIN_VOLUME           = 500_000
DAYS_TO_REMOVE       = 2   # дней без сигнала → удаляем из списка


# ─── Загрузка вселенной акций ────────────────────────────────────────────────

def download_us_universe() -> list[str]:
    """Скачивает все тикеры NYSE + NASDAQ с nasdaqtrader.com."""
    urls = {
        "nasdaq": "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        "other":  "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    }
    tickers = set()
    for name, url in urls.items():
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep='|')
            sym_col = 'Symbol' if 'Symbol' in df.columns else 'ACT Symbol'
            if sym_col not in df.columns:
                continue
            etf_col = 'ETF' if 'ETF' in df.columns else None
            for _, row in df.iterrows():
                sym = str(row[sym_col]).strip()
                if etf_col and str(row.get(etf_col, '')).upper() == 'Y':
                    continue
                if not sym.replace('-', '').isalpha():
                    continue
                if len(sym) > 5 or len(sym) < 1:
                    continue
                # Исключаем warrants (W), units (U), rights (R) в конце
                if sym[-1] in ('W', 'U', 'R') and len(sym) > 3:
                    continue
                tickers.add(sym)
        except Exception as e:
            print(f"[SCREENER] Universe download error ({name}): {e}")
    return list(tickers)


# ─── Предфильтр по цене и объёму ─────────────────────────────────────────────

def batch_prefilter(tickers: list[str], yf) -> list[str]:
    """Оставляет только акции с ценой > $5 и объёмом > 500K (5-дневный батч)."""
    valid = []
    batch_size = 200
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        try:
            data = yf.download(batch, period="5d", interval="1d",
                               progress=False, auto_adjust=False)
            for ticker in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        close  = data['Close'][ticker].dropna()
                        volume = data['Volume'][ticker].dropna()
                    else:
                        close  = data['Close'].dropna()
                        volume = data['Volume'].dropna()
                    if close.empty or volume.empty:
                        continue
                    if float(close.iloc[-1]) >= MIN_PRICE and float(volume.mean()) >= MIN_VOLUME:
                        valid.append(ticker)
                except Exception:
                    continue
        except Exception as e:
            print(f"[SCREENER] Pre-filter batch error: {e}")
        time.sleep(0.5)
    return valid


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _calc_emas(hist: pd.DataFrame) -> pd.DataFrame:
    for p in [10, 20, 50, 200]:
        hist[f'ema_{p}'] = hist['Close'].ewm(span=p, adjust=False).mean()
    # ATR (14-period)
    hi = hist['High']
    lo = hist['Low']
    pr = hist['Close'].shift(1)
    tr = pd.concat([hi - lo, (hi - pr).abs(), (lo - pr).abs()], axis=1).max(axis=1)
    hist['atr'] = tr.ewm(span=14, adjust=False).mean()
    return hist


def _ema200_has_bounce_history(hist: pd.DataFrame, min_touches: int = 2) -> bool:
    """
    EMA200 выступала как поддержка >= min_touches раз в истории:
    - цена входила в зону ±1 ATR от EMA200 сверху
    - не закрывалась ниже EMA200 - 1 ATR
    - отскакивала выше EMA200 + 1 ATR с ростом >= 1 ATR от минимума касания
    """
    if len(hist) < 60 or 'atr' not in hist.columns:
        return False

    n       = len(hist) - 1  # исключаем текущий (последний) бар
    closes  = hist['Close'].values
    ema200s = hist['ema_200'].values
    atrs    = hist['atr'].values

    touches   = 0
    in_touch  = False
    touch_low = 0.0
    touch_atr = 0.0
    was_above = False  # price was above EMA200 + 1 ATR on previous bar

    for i in range(n):
        try:
            close  = float(closes[i])
            ema200 = float(ema200s[i])
            atr    = float(atrs[i])
            if ema200 <= 0 or atr <= 0:
                continue

            above_zone = close > ema200 + atr
            in_zone    = ema200 - atr <= close <= ema200 + atr
            below_zone = close < ema200 - atr

            if not in_touch:
                if was_above and in_zone:
                    in_touch  = True
                    touch_low = close
                    touch_atr = atr
            else:
                if below_zone:
                    in_touch = False
                elif above_zone:
                    if (close - touch_low) >= touch_atr:
                        touches += 1
                    in_touch  = False
                    was_above = True
                    continue
                else:
                    if close < touch_low:
                        touch_low = close

            was_above = above_zone
        except Exception:
            continue

    return touches >= min_touches


def _is_hammer(high, low, open_, close) -> bool:
    total = high - low
    if total < 0.001:
        return False
    body       = abs(close - open_)
    lower_wick = min(close, open_) - low
    upper_wick = high - max(close, open_)
    body = max(body, total * 0.005)
    return (lower_wick >= 2.0 * body and
            upper_wick <= body and
            body / total <= 0.35)


# ─── Критерии скрининга ───────────────────────────────────────────────────────

def screen_ready_20ema(hist: pd.DataFrame) -> bool:
    """
    Цена между EMA10 и EMA20 минимум 2 свечи подряд.
    Выше 50 EMA и 200 EMA. EMA20 выше EMA50. Расстояние от EMA20 не более 1 ATR.
    """
    if len(hist) < 5:
        return False
    cur  = float(hist['Close'].iloc[-1])
    e10c = float(hist['ema_10'].iloc[-1])
    e20c = float(hist['ema_20'].iloc[-1])
    e50c = float(hist['ema_50'].iloc[-1])
    if cur < e50c or cur < float(hist['ema_200'].iloc[-1]):
        return False
    # EMA20 ниже EMA50 — тренд не восходящий, подхват не актуален (NEE и подобные)
    if e20c < e50c:
        return False
    # Цена уже ушла выше EMA10 более чем на 0.5% — сигнал устарел, точки входа нет
    if cur > e10c * 1.005:
        return False
    # Цена дальше 1 ATR от EMA20 — слишком далеко для подхвата
    if 'atr' in hist.columns and e20c > 0:
        atr = float(hist['atr'].iloc[-1])
        if abs(cur - e20c) > atr:
            return False

    consecutive = 0
    for i in range(-1, -8, -1):
        try:
            c   = float(hist['Close'].iloc[i])
            e10 = float(hist['ema_10'].iloc[i])
            e20 = float(hist['ema_20'].iloc[i])
            if c < e20 * 0.98:          # нарушение — закрылась ниже EMA20 на >2%
                return False
            if e20 * 0.98 <= c <= e10 * 1.02:
                consecutive += 1
            else:
                break
        except Exception:
            break
    return consecutive >= 2


def screen_swing_20ema(hist: pd.DataFrame) -> bool:
    """
    Цена 2+ раза отскакивала от EMA20 пока растёт.
    Сейчас снова у EMA20. EMA20 растущая.
    """
    if len(hist) < 30:
        return False
    cur = float(hist['Close'].iloc[-1])
    if cur < float(hist['ema_50'].iloc[-1]) or cur < float(hist['ema_200'].iloc[-1]):
        return False
    if float(hist['ema_20'].iloc[-1]) <= float(hist['ema_20'].iloc[-11]):
        return False

    touches, in_touch = 0, False
    for i in range(-30, 0):
        try:
            e20  = float(hist['ema_20'].iloc[i])
            dist = abs(float(hist['Close'].iloc[i]) - e20) / e20 * 100
            if dist <= 1.5 and not in_touch:
                touches += 1
                in_touch = True
            elif dist > 3.0:
                in_touch = False
        except Exception:
            continue

    if touches < 2:
        return False
    curr_dist = abs(cur - float(hist['ema_20'].iloc[-1])) / float(hist['ema_20'].iloc[-1]) * 100
    return curr_dist <= 2.0


def screen_first_touch_20ema(hist: pd.DataFrame) -> bool:
    """
    Цена резко выросла (20%+ за 30 баров), теперь первый раз касается EMA20.
    """
    if len(hist) < 35:
        return False
    close  = hist['Close']
    ema20  = hist['ema_20']
    cur    = float(close.iloc[-1])
    e20_cur = float(ema20.iloc[-1])

    # Текущая цена у EMA20
    if abs(cur - e20_cur) / e20_cur * 100 > 3.0:
        return False

    # Был резкий рост: минимум за последние 30 баров vs максимум за 20 баров
    low_30  = float(close.iloc[-31:-1].min())
    high_20 = float(close.iloc[-20:].max())
    if low_30 <= 0 or (high_20 - low_30) / low_30 * 100 < 20:
        return False

    # Не было предыдущих касаний EMA20 в последние 30 баров (кроме текущего)
    prev_touches = sum(
        1 for i in range(-30, -1)
        if abs(float(close.iloc[i]) - float(ema20.iloc[i])) / float(ema20.iloc[i]) * 100 <= 2.0
    )
    return prev_touches == 0


def screen_two_hammers_weekly(hist_w: pd.DataFrame) -> bool:
    """Два молота подряд на недельном таймфрейме."""
    if len(hist_w) < 3:
        return False
    for i in (-1, -2):
        row = hist_w.iloc[i]
        if not _is_hammer(float(row['High']), float(row['Low']),
                          float(row['Open']), float(row['Close'])):
            return False
    return True


def screen_three_hammers(hist: pd.DataFrame) -> bool:
    """Три молота подряд на любом таймфрейме. Без дополнительных условий."""
    if len(hist) < 4:
        return False
    for i in (-1, -2, -3):
        row = hist.iloc[i]
        if not _is_hammer(float(row['High']), float(row['Low']),
                          float(row['Open']), float(row['Close'])):
            return False
    return True


def screen_double_triple_strike(hist: pd.DataFrame) -> dict | None:
    """
    Текущий бар поглощает диапазон 2 (двойной) или 3 (тройной) предыдущих.
    Сигнал разворота после тренда 15%+.
    """
    if len(hist) < 5:
        return None
    tol = 0.015
    h0, l0 = float(hist['High'].iloc[-1]), float(hist['Low'].iloc[-1])
    h1, l1 = float(hist['High'].iloc[-2]), float(hist['Low'].iloc[-2])
    h2, l2 = float(hist['High'].iloc[-3]), float(hist['Low'].iloc[-3])

    double = (h0 >= h1 * (1 - tol) and h0 >= h2 * (1 - tol) and
              l0 <= l1 * (1 + tol) and l0 <= l2 * (1 + tol))
    if not double:
        return None

    triple = False
    if len(hist) >= 5:
        h3, l3 = float(hist['High'].iloc[-4]), float(hist['Low'].iloc[-4])
        triple = h0 >= h3 * (1 - tol) and l0 <= l3 * (1 + tol)

    # Проверяем предшествующий тренд (15%+ за 40 баров)
    lookback = min(40, len(hist) - 1)
    start    = float(hist['Close'].iloc[-(lookback + 1)])
    end      = float(hist['Close'].iloc[-2])
    trend    = (end - start) / start * 100 if start > 0 else 0
    if abs(trend) < 15:
        return None

    return {
        'type':      'triple' if triple else 'double',
        'direction': 'bearish' if trend > 0 else 'bullish',
        'trend_pct': round(trend, 1),
    }


def screen_ema_approach_daily(hist: pd.DataFrame) -> list[str]:
    """EMA 20/50/200 подход сверху на дневном (упрощённо для скринера)."""
    found = []
    cur        = float(hist['Close'].iloc[-1])
    ema200_val = float(hist['ema_200'].iloc[-1])
    if cur < ema200_val:
        return found
    ema20_val = float(hist['ema_20'].iloc[-1])
    ema50_val = float(hist['ema_50'].iloc[-1])
    for period, entry_pct, approach_pct in [(20, 1.0, 2.0), (50, 1.5, 3.0), (200, 4.0, 6.0)]:
        col = f'ema_{period}'
        if col not in hist.columns:
            continue
        ema_val = float(hist[col].iloc[-1])
        if cur < ema_val:
            continue
        dist = (cur - ema_val) / ema_val * 100
        came_from_above = all(
            float(hist['Close'].iloc[-i]) >= float(hist[col].iloc[-i])
            for i in range(2, 5) if len(hist) > i
        )
        if not came_from_above:
            continue
        # EMA200: восходящий тренд (EMA20 > EMA50 > EMA200) + минимум 2 исторических отскока
        if period == 200:
            if not (ema20_val > ema50_val > ema200_val):
                continue
            if not _ema200_has_bounce_history(hist, min_touches=2):
                continue
        if dist <= entry_pct:
            found.append(f'ema{period}_daily_entry')
        elif dist <= approach_pct:
            found.append(f'ema{period}_daily_approach')
    return found


def screen_ema_approach_weekly(hist_w: pd.DataFrame) -> list[str]:
    """EMA 20/50/200 подход сверху на недельном."""
    found = []
    if len(hist_w) < 5:
        return found
    cur        = float(hist_w['Close'].iloc[-1])
    ema200_val = float(hist_w['ema_200'].iloc[-1])
    if cur < ema200_val:
        return found
    ema20_val = float(hist_w['ema_20'].iloc[-1])
    ema50_val = float(hist_w['ema_50'].iloc[-1])
    for period, entry_pct, approach_pct in [(20, 2.0, 4.0), (50, 2.0, 4.0), (200, 5.0, 8.0)]:
        col = f'ema_{period}'
        if col not in hist_w.columns:
            continue
        ema_val = float(hist_w[col].iloc[-1])
        if cur < ema_val:
            continue
        dist = (cur - ema_val) / ema_val * 100
        came_from_above = all(
            float(hist_w['Close'].iloc[-i]) >= float(hist_w[col].iloc[-i])
            for i in range(2, 4) if len(hist_w) > i
        )
        if not came_from_above:
            continue
        # EMA200: восходящий тренд (EMA20 > EMA50 > EMA200) + минимум 2 исторических отскока
        if period == 200:
            if not (ema20_val > ema50_val > ema200_val):
                continue
            if not _ema200_has_bounce_history(hist_w, min_touches=2):
                continue
        if dist <= entry_pct:
            found.append(f'ema{period}_weekly_entry')
        elif dist <= approach_pct:
            found.append(f'ema{period}_weekly_approach')
    return found


# ─── Батч-скрининг ───────────────────────────────────────────────────────────

def _screen_one(ticker: str, hist_d: pd.DataFrame, hist_w: pd.DataFrame) -> dict | None:
    """Применяет все критерии к одному тикеру."""
    if len(hist_d) < 50:
        return None
    try:
        cur = float(hist_d['Close'].iloc[-1])
        avg_vol = float(hist_d['Volume'].iloc[-20:].mean())
        if cur < MIN_PRICE or avg_vol < MIN_VOLUME:
            return None

        hist_d = _calc_emas(hist_d)
        if len(hist_w) >= 20:
            hist_w = _calc_emas(hist_w)

        found: dict[str, object] = {}

        # EMA подходы
        for sig in screen_ema_approach_daily(hist_d):
            found[sig] = True
        if len(hist_w) >= 20:
            for sig in screen_ema_approach_weekly(hist_w):
                found[sig] = True

        if screen_ready_20ema(hist_d):
            found['ready_20ema'] = True
        if screen_swing_20ema(hist_d):
            found['swing_20ema'] = True
        if screen_first_touch_20ema(hist_d):
            found['first_touch_20ema'] = True
        if len(hist_w) >= 10 and screen_two_hammers_weekly(hist_w):
            found['two_hammers_weekly'] = True
        if screen_two_hammers_weekly(hist_d):
            found['two_hammers_daily'] = True
        if screen_three_hammers(hist_d):
            found['three_hammers_daily'] = True
        if len(hist_w) >= 4 and screen_three_hammers(hist_w):
            found['three_hammers_weekly'] = True

        strike_d = screen_double_triple_strike(hist_d)
        if strike_d:
            found['strike_daily'] = strike_d
        if len(hist_w) >= 5:
            strike_w = screen_double_triple_strike(hist_w)
            if strike_w:
                found['strike_weekly'] = strike_w

        if not found:
            return None

        return {
            'ticker':       ticker,
            'price':        round(cur, 2),
            'signal_types': [k for k in found if found[k]],
            'signals':      found,
            'date':         datetime.now().strftime('%Y-%m-%d'),
        }
    except Exception:
        return None


def screen_batch(tickers: list[str], yf) -> dict:
    """Батч-скрининг: скачиваем данные группами по 50 и применяем критерии."""
    results = {}
    batch_size = 50
    total = len(tickers)

    # Батч дневной (1 год)
    daily_batches  = {}
    for i in range(0, total, batch_size):
        batch = tickers[i: i + batch_size]
        try:
            data = yf.download(batch, period="1y", interval="1d",
                               progress=False, auto_adjust=False)
            for t in batch:
                try:
                    h = data.xs(t, level=1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data
                    daily_batches[t] = h.dropna()
                except Exception:
                    pass
        except Exception as e:
            print(f"[SCREENER] Daily batch error: {e}")
        time.sleep(1)
        if (i // batch_size + 1) % 5 == 0:
            print(f"[SCREENER] Daily download: {min(i+batch_size, total)}/{total}")

    # Батч недельный (2 года)
    weekly_batches = {}
    for i in range(0, total, batch_size):
        batch = tickers[i: i + batch_size]
        try:
            data = yf.download(batch, period="2y", interval="1wk",
                               progress=False, auto_adjust=False)
            for t in batch:
                try:
                    h = data.xs(t, level=1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data
                    weekly_batches[t] = h.dropna()
                except Exception:
                    pass
        except Exception as e:
            print(f"[SCREENER] Weekly batch error: {e}")
        time.sleep(1)

    # Применяем критерии
    for t in tickers:
        h_d = daily_batches.get(t, pd.DataFrame())
        h_w = weekly_batches.get(t, pd.DataFrame())
        r = _screen_one(t, h_d, h_w)
        if r:
            results[t] = r

    return results


# ─── Управление активным списком тикеров ─────────────────────────────────────

def load_active_tickers() -> dict:
    if ACTIVE_TICKERS_PATH.exists():
        try:
            return json.loads(ACTIVE_TICKERS_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def save_active_tickers(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_TICKERS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, default=str, indent=2),
        encoding='utf-8'
    )


def update_active_tickers(screener_results: dict) -> dict:
    """Добавляет новые тикеры с сигналами, удаляет без сигналов 2+ дня."""
    active = load_active_tickers()
    today  = datetime.now().strftime('%Y-%m-%d')
    cutoff = datetime.now() - timedelta(days=DAYS_TO_REMOVE)

    for ticker, result in screener_results.items():
        active[ticker] = {
            'added':        active.get(ticker, {}).get('added', today),
            'last_signal':  today,
            'signal_types': result.get('signal_types', []),
            'price':        result.get('price'),
            'category':     'Screener',
        }

    to_remove = []
    for t, d in active.items():
        try:
            last = datetime.strptime(d.get('last_signal', '2000-01-01'), '%Y-%m-%d')
            if last < cutoff:
                to_remove.append(t)
        except Exception:
            pass

    for t in to_remove:
        del active[t]
        print(f"[SCREENER] Removed {t} (no signal {DAYS_TO_REMOVE}+ days)")

    save_active_tickers(active)
    print(f"[SCREENER] Active tickers: {len(active)} (+{len(screener_results)} found, -{len(to_remove)} removed)")
    return active


# ─── Главная функция ─────────────────────────────────────────────────────────

def run_full_screener(yf) -> dict:
    """
    Полный скан всех акций США.
    1. Скачиваем вселенную (~7000 тикеров)
    2. Предфильтр по цене и объёму (~2000 остаётся)
    3. Батч-скрининг по всем критериям
    4. Обновляем active_tickers.json
    """
    t0 = time.time()
    print("[SCREENER] === Starting full universe scan ===")

    universe = download_us_universe()
    print(f"[SCREENER] Universe: {len(universe)} tickers")

    valid = batch_prefilter(universe, yf)
    print(f"[SCREENER] After pre-filter (price>${MIN_PRICE}, vol>{MIN_VOLUME:,}): {len(valid)} tickers")

    results = screen_batch(valid, yf)
    print(f"[SCREENER] Signals found: {len(results)} tickers")

    active = update_active_tickers(results)

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output = {
        'last_update':   now_str,
        'total_scanned': len(valid),
        'total_found':   len(results),
        'results':       results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCREENER_RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, default=str),
        encoding='utf-8'
    )

    elapsed = int(time.time() - t0)
    print(f"[SCREENER] Done in {elapsed//60}m {elapsed%60}s. Active: {len(active)}")
    return output
