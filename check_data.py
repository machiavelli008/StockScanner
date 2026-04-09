"""
Показывает сырые недельные данные MSFT за Jul-Oct 2024
Запуск: python check_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
from backend.main import calculate_atr, calculate_ema
from datetime import datetime, timedelta

raw = yf.download("MSFT", period="15y", interval="1wk", auto_adjust=False, progress=False)

import pandas as pd
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

hist = raw[['Open','High','Low','Close','Volume']].copy()
hist.dropna(subset=['Close'], inplace=True)
hist = calculate_atr(hist)
for p in [20, 50]:
    hist[f'ema_{p}'] = calculate_ema(hist['Close'], p)

start = pd.Timestamp('2024-07-01')
end   = pd.Timestamp('2024-10-07')
subset = hist[(hist.index >= start) & (hist.index <= end)]

print(f"\n{'Дата':<12} {'Close':>7} {'EMA20':>7} {'ATR':>6} {'EMA-ATR':>9} {'EMA+1%':>9} {'EMA50':>7}")
print("-" * 65)
for dt, row in subset.iterrows():
    e20 = row['ema_20']
    e50 = row['ema_50']
    a   = row['atr']
    c   = row['Close']
    flag = " ← POSITIVE?" if c >= e20 * 1.01 else (" ← NEG?" if c < e20 - a else "")
    print(f"  {pd.Timestamp(dt).strftime('%Y-%m-%d')}  {c:>7.2f}  {e20:>7.2f}  {a:>6.2f}  {e20-a:>9.2f}  {e20*1.01:>9.2f}  {e50:>7.2f}{flag}")
