import os
import json
import random
import pandas as pd


PROMPT = """Ты анализируешь данные касаний EMA для алгоритма сканера акций.

Правила классификации:
- POSITIVE: цена закрылась >= EMA + 1ATR в течение 12 баров после касания
- NEGATIVE: цена закрылась < EMA - 1ATR, ИЛИ 3 закрытия подряд ниже EMA (weekly)
- Касание: цена приходит сверху (3+ бара выше EMA) и close в 0.15% от EMA ИЛИ low касается EMA

Данные ({count} акций, EMA {ema} {timeframe}, последние 5 лет):
{data}

Найди подозрительные случаи:
1. 100% позитивных при 3+ касаниях — возможно порог слишком мягкий
2. 0% позитивных при 3+ касаниях — возможно логика не работает
3. Касания ближе 3 баров друг к другу — возможно дублирование
4. ATR аномально маленький или большой относительно цены

Отвечай на русском. Только реально подозрительные случаи.
Формат: ТИКЕР (дата): причина
Не более 400 символов. Если всё нормально — напиши "ок"."""


def _format_batch(batch):
    lines = []
    for item in batch:
        ticker = item['ticker']
        p = item['positive']
        total = item['total']
        if total == 0:
            lines.append(f"{ticker}: нет касаний")
            continue
        touches = ", ".join(
            f"{t['date']}({'+'if t['result']=='positive'else'-'}) p={t['price']} e={t['ema']} atr={t['atr']}"
            for t in item['touches']
        )
        lines.append(f"{ticker}: {p}/{total} | {touches}")
    return "\n".join(lines)


def _analyze_with_claude(batch, ema_period, timeframe, api_key):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = PROMPT.format(
            count=len(batch),
            ema=ema_period,
            timeframe=timeframe,
            data=_format_batch(batch)
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        if text.lower().startswith("ок") or text.lower() == "ok":
            return None
        return text
    except Exception as e:
        return f"Ошибка API: {e}"


def _collect_touches(ticker, ema_period, timeframe, yf,
                     find_touch_fn, calc_ema_fn, calc_atr_fn, normalize_fn):
    try:
        hist = yf.download(
            ticker, period="10y",
            interval="1wk" if timeframe == "weekly" else "1d",
            progress=False, auto_adjust=False
        )
        hist = normalize_fn(hist)
        if hist.empty or len(hist) < 50:
            return None

        for p in [10, 20, 50, 100, 200]:
            hist[f'ema_{p}'] = calc_ema_fn(hist['Close'], p)
        hist = calc_atr_fn(hist, 14)
        hist = hist.dropna()
        hist.index = pd.to_datetime(hist.index)

        cutoff = hist.index.max() - pd.DateOffset(years=5)
        hist = hist[hist.index > cutoff]

        ema_col = f'ema_{ema_period}'
        all_cols = ['ema_20', 'ema_50', 'ema_100', 'ema_200']
        lower = [] if ema_period == 20 else [c for c in all_cols if c != ema_col]

        if timeframe == "weekly":
            touches = find_touch_fn(
                hist, ema_col, 'atr',
                lower_ema_cols=lower if lower else None,
                cooldown_bars=2,
                require_rally_after_negative=True,
                min_ema_slope_bars=8 if ema_period <= 50 else 0,
                min_ema_slope_pct=0.0 if ema_period == 20 else 0.001,
            )
        else:
            touches = find_touch_fn(
                hist, ema_col, 'atr',
                lower_ema_cols=lower if lower else None,
                cooldown_bars=0,
                min_ema_slope_pct=0.0 if ema_period == 20 else 0.001,
            )

        positive = sum(1 for t in touches if t['result'] == 'positive')
        return {
            'ticker': ticker,
            'positive': positive,
            'negative': len(touches) - positive,
            'total': len(touches),
            'touches': [
                {
                    'date':   str(t['date'])[:10],
                    'result': t['result'],
                    'price':  round(float(t['price']), 2),
                    'ema':    round(float(t['ema']), 2),
                    'atr':    round(float(t['atr']), 2),
                }
                for t in touches
            ]
        }
    except Exception:
        return None


def run_analysis(signals_path, find_touch_fn, calc_ema_fn, calc_atr_fn,
                 normalize_fn, get_yf_fn, n_tickers=10):
    """
    Берёт n_tickers случайных акций, проверяет EMA 20/50/200 на daily и weekly.
    Возвращает список найденных аномалий.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return ["ANTHROPIC_API_KEY не задан — агент не запущен"]

    try:
        with open(signals_path) as f:
            data = json.load(f)
        signals = data.get("signals", data) if isinstance(data, dict) else data
    except Exception as e:
        return [f"Ошибка чтения signals.json: {e}"]

    all_tickers = [s['ticker'] for s in signals if s.get('ticker')]
    if not all_tickers:
        return ["Нет тикеров для анализа"]

    selected = random.sample(all_tickers, min(n_tickers, len(all_tickers)))
    yf = get_yf_fn()

    findings = []

    for ema_period in [20, 50, 200]:
        for timeframe in ['daily', 'weekly']:
            batch = []
            for ticker in selected:
                result = _collect_touches(
                    ticker, ema_period, timeframe, yf,
                    find_touch_fn, calc_ema_fn, calc_atr_fn, normalize_fn
                )
                if result:
                    batch.append(result)

            if not batch:
                continue

            analysis = _analyze_with_claude(batch, ema_period, timeframe, api_key)
            if analysis:
                findings.append(f"EMA{ema_period} {timeframe}:\n{analysis}")

    return findings
