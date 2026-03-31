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
from backend.main import load_tickers, get_stock_signals
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

if __name__ == "__main__":
    main()
