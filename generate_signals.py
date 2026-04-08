"""
Запускай этот скрипт локально (на своём компьютере) когда хочешь обновить данные.
Он скачает данные с Yahoo Finance и сохранит в data/signals.json
Потом сделай git push — Railway автоматически покажет новые данные.

Запуск:  python generate_signals.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from pathlib import Path
from datetime import datetime
from backend.main import load_tickers, get_stock_signals, calculate_atr, calculate_ema
import time

OUTPUT_PATH = Path(__file__).parent / "data" / "signals.json"

def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    tickers = load_tickers()
    print(f"\nFound {len(tickers)} tickers: {', '.join(tickers)}")
    print("=" * 50)

    signals = []
    failed = []

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(2)
        print(f"\n[{i+1}/{len(tickers)}] Processing {ticker}...")
        for attempt in range(3):
            result = get_stock_signals(ticker)
            if result is not None:
                signals.append(result)
                break
            if attempt < 2:
                print(f"  Retrying in 10s...")
                time.sleep(10)
        else:
            print(f"  FAILED after 3 attempts: {ticker}")
            failed.append(ticker)

    output = {
        "signals": signals,
        "last_update": datetime.now().isoformat(),
        "total": len(signals),
        "failed": failed,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, default=str)

    print("\n" + "=" * 50)
    print(f"Done! {len(signals)} signals saved to {OUTPUT_PATH}")
    if failed:
        print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")
    print("\nNext step: git add data/signals.json && git commit -m 'Update signals' && git push")

def check_touches(ticker="MSFT", ema_period=20):
    import sys, pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from backend.main import get_yfinance, normalize_ohlc_columns, calculate_atr, calculate_ema, find_touch_events
    from datetime import datetime, timedelta
    yf = get_yfinance()
    raw = yf.download(ticker, period="15y", interval="1wk", auto_adjust=False, progress=False)
    raw = normalize_ohlc_columns(raw)
    hist = raw[['Open','High','Low','Close','Volume']].copy()
    hist.dropna(subset=['Close'], inplace=True)
    hist = calculate_atr(hist)
    for p in [20, 50, 100, 200]:
        hist[f'ema_{p}'] = calculate_ema(hist['Close'], p)
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=365*10))
    hist = hist[hist.index > cutoff]
    all_periods = [20, 50, 100, 200]
    ema_col = f'ema_{ema_period}'
    lower_cols = [f'ema_{p}' for p in all_periods if p != ema_period]
    touches = find_touch_events(hist, ema_col, 'atr', lower_ema_cols=lower_cols,
                                cooldown_bars=2, require_rally_after_negative=True,
                                max_bars_below_ema=3 if ema_period <= 50 else None)
    pos = [t for t in touches if t['result'] == 'positive']
    neg = [t for t in touches if t['result'] == 'negative']
    print(f"\n{ticker} WEEKLY EMA{ema_period}:")
    print(f"ПОЗИТИВНЫЕ ({len(pos)}):")
    for t in pos:
        print(f"  {pd.Timestamp(t['date']).strftime('%Y-%m-%d')}  price={t['price']:.2f}  ema={t['ema']:.2f}")
    print(f"НЕГАТИВНЫЕ ({len(neg)}):")
    for t in neg:
        print(f"  {pd.Timestamp(t['date']).strftime('%Y-%m-%d')}  price={t['price']:.2f}  ema={t['ema']:.2f}")
    print(f"ИТОГО: {len(pos)} up | {len(neg)} down")

def check_raw_data():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from backend.main import get_yfinance, normalize_ohlc_columns
    import pandas as pd
    yf = get_yfinance()
    raw = yf.download("MSFT", period="15y", interval="1wk", auto_adjust=False, progress=False)
    raw = normalize_ohlc_columns(raw)
    hist = raw[['Open','High','Low','Close','Volume']].copy()
    hist.dropna(subset=['Close'], inplace=True)
    hist = calculate_atr(hist)
    hist['ema_20'] = calculate_ema(hist['Close'], 20)
    hist['ema_50'] = calculate_ema(hist['Close'], 50)
    start = pd.Timestamp('2024-07-01')
    end   = pd.Timestamp('2024-10-07')
    subset = hist[(hist.index >= start) & (hist.index <= end)]
    print(f"\n{'Дата':<12} {'Close':>7} {'EMA20':>7} {'ATR':>6} {'EMA-ATR':>9} {'EMA+1%':>9} {'EMA50':>7}")
    print("-" * 65)
    for dt, row in subset.iterrows():
        e = row['ema_20']; a = row['atr']; c = row['Close']; e50 = row['ema_50']
        flag = " ← POS?" if c >= e*1.01 else (" ← NEG?" if c < e-a else "")
        print(f"  {pd.Timestamp(dt).strftime('%Y-%m-%d')}  {c:>7.2f}  {e:>7.2f}  {a:>6.2f}  {e-a:>9.2f}  {e*1.01:>9.2f}  {e50:>7.2f}{flag}")

if __name__ == "__main__":
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "check":
        check_raw_data()
    elif len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "touches":
        ticker = __import__('sys').argv[2] if len(__import__('sys').argv) > 2 else "MSFT"
        period = int(__import__('sys').argv[3]) if len(__import__('sys').argv) > 3 else 20
        check_touches(ticker, period)
    else:
        main()
