"""
Показывает даты отдельных касаний EMA для отладки.
Запуск: python debug_touches.py MSFT weekly 20
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from backend.main import calculate_atr, calculate_ema, find_touch_events

def debug_touches(ticker, timeframe, ema_period):
    ema_periods = [20, 50, 100, 200]
    print(f"\n{'='*60}")
    print(f"  {ticker} | {timeframe.upper()} | EMA{ema_period}")
    print(f"{'='*60}")

    # Скачиваем 15 лет данных
    raw = yf.download(ticker, period="15y", interval="1wk" if timeframe == "weekly" else "1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        print("Нет данных")
        return

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    hist = raw[['Open','High','Low','Close','Volume']].copy()
    hist.dropna(subset=['Close'], inplace=True)

    # Считаем ATR и EMA
    hist = calculate_atr(hist)
    for p in ema_periods:
        hist[f'ema_{p}'] = calculate_ema(hist['Close'], p)

    # Обрезаем до 10 лет
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=365 * 10))
    hist = hist[hist.index > cutoff]

    ema_col = f'ema_{ema_period}'
    lower_cols = [f'ema_{p}' for p in ema_periods if p > ema_period]
    cooldown = 5 if timeframe == "weekly" else 0

    touches = find_touch_events(
        hist, ema_col, 'atr',
        lookahead=12,
        near_pct=0.0015,
        rebound_pct=0.01,
        lower_ema_cols=lower_cols,
        cooldown_bars=cooldown,
        require_rally_after_negative=(timeframe == "weekly"),
        max_bars_below_ema=3 if timeframe == "weekly" else None,
    )

    positives = [t for t in touches if t['result'] == 'positive']
    negatives = [t for t in touches if t['result'] == 'negative']

    print(f"\nПОЗИТИВНЫЕ ({len(positives)}):")
    for t in positives:
        date_str = pd.Timestamp(t['date']).strftime('%Y-%m-%d')
        print(f"  {date_str}  price={t['price']:.2f}  ema={t['ema']:.2f}")

    print(f"\nНЕГАТИВНЫЕ ({len(negatives)}):")
    for t in negatives:
        date_str = pd.Timestamp(t['date']).strftime('%Y-%m-%d')
        print(f"  {date_str}  price={t['price']:.2f}  ema={t['ema']:.2f}")

    print(f"\nИТОГО: {len(positives)} up | {len(negatives)} down")

    # Сырые данные за Jul-Oct 2024 для проверки
    print(f"\n{'='*60}")
    print("СЫРЫЕ ДАННЫЕ Jul-Oct 2024:")
    print(f"  {'Дата':<12} {'Close':>7} {'EMA':>7} {'ATR':>6} {'EMA-ATR':>9} {'EMA+1%':>9}")
    target = pd.Timestamp('2024-07-01')
    end    = pd.Timestamp('2024-10-07')
    subset = hist[(hist.index >= target) & (hist.index <= end)]
    for dt, row in subset.iterrows():
        e = row[ema_col]
        a = row['atr']
        c = row['Close']
        print(f"  {pd.Timestamp(dt).strftime('%Y-%m-%d')}  {c:>7.2f}  {e:>7.2f}  {a:>6.2f}  {e-a:>9.2f}  {e*1.01:>9.2f}")

if __name__ == "__main__":
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "weekly"
    period   = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    debug_touches(ticker, timeframe, period)
