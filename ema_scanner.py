import yfinance as yf
import pandas as pd

# Файл с тикерами
tickers_file = "tickers.csv"
tickers = pd.read_csv(tickers_file)['Ticker'].tolist()

# Параметры EMA и анализа касаний
ema_periods = [20, 50, 100, 200]
threshold_pct = 0.005  # 0.5% для определения касания
lookahead_days = 5     # Сколько дней смотрим после касания

# Результаты по каждой EMA
all_results = {f'EMA{p}': [] for p in ema_periods}

for ticker in tickers:
    try:
        # Загружаем 10 лет данных
        data = yf.download(ticker, period="10y", interval="1d")
        if data.empty:
            print(f"Данных нет для {ticker}, пропускаем")
            continue

        # Вычисляем EMA
        for period in ema_periods:
            data[f'EMA{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

            touches = 0
            bounces = 0
            breaks = 0

            for i in range(len(data)):
                price = data['Close'].iloc[i]
                ema = data[f'EMA{period}'].iloc[i]

                if pd.isna(ema):
                    continue

                # Проверяем касание EMA (только скалярные значения)
                if abs(float(price) - float(ema)) / float(ema) <= threshold_pct:
                    touches += 1
                    future = data['Close'].iloc[i+1:i+1+lookahead_days]
                    if future.empty:
                        continue
                    # Отскок или пробой
                    if (future > ema).any():
                        bounces += 1
                    else:
                        breaks += 1

            if touches > 0:
                prob_bounce = round((bounces / touches) * 100, 2)
                all_results[f'EMA{period}'].append({
                    'Ticker': ticker,
                    'Touches': touches,
                    'Bounces': bounces,
                    'Breaks': breaks,
                    'Prob_Bounce (%)': prob_bounce
                })

        print(f"{ticker} обработан ✅")

    except Exception as e:
        print(f"Ошибка с тикером {ticker}: {e}")

# Сохраняем Excel с листами для каждой EMA
with pd.ExcelWriter('ema_touch_analysis_fixed.xlsx') as writer:
    for period in ema_periods:
        df = pd.DataFrame(all_results[f'EMA{period}'])
        df.to_excel(writer, sheet_name=f'EMA{period}', index=False)

print("Готово! Файл ema_touch_analysis_fixed.xlsx создан.")