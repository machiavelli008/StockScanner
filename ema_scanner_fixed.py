import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Полный путь к CSV
tickers_file = r"C:\Users\Administrator\StockScanner\tickers.csv"

if not os.path.exists(tickers_file):
    print(f"Файл не найден: {tickers_file}")
    exit(1)
else:
    print(f"Файл найден: {tickers_file}")

tickers = pd.read_csv(tickers_file)['Ticker'].tolist()
print(f"Загружено тикеров: {len(tickers)}\n")

ema_periods = [20, 50, 100, 150, 200]
timeframes = ['1d', '1wk']
atr_period = 14
lookahead_days = 5
min_total_interactions = 2

all_results = {}
touch_details = {}

for tf in timeframes:
    all_results[tf] = {f'EMA{p}': [] for p in ema_periods}
    touch_details[tf] = {f'EMA{p}': [] for p in ema_periods}

def calculate_atr(data, period=14):
    """Расчет ATR"""
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift()),
            abs(data['low'] - data['close'].shift())
        )
    )
    data['atr'] = data['tr'].rolling(window=period).mean()
    return data

def find_touch_events(data, ema_col, atr_col, lookahead):
    """
    ✅ ПРАВИЛЬНАЯ ЛОГИКА:
    
    1. КАСАНИЕ: Цена подходит к EMA (±0.15%) из верхней зоны (выше EMA)
    2. ГРУППА СВЕЧЕЙ: Несколько дней подряд в зоне = 1 взаимодействие
    3. ПОЛОЖИТЕЛЬНОЕ: Пробила вниз, но выше EMA-1ATR + потом ПОШЛА ВЫШЕ EMA (подтверждение)
    4. ОТРИЦАТЕЛЬНОЕ: Закрытие ниже EMA-1ATR (полное пробитие)
    5. НЕЙТРАЛЬНОЕ: Консолидация без явного разворота вверх
    6. 1 КАСАНИЕ = 1 ГРУППА СВЕЧЕЙ = 1 СОБЫТИЕ
    """
    
    touches = []
    processed_indices = set()
    
    for i in range(1, len(data) - lookahead - 1):
        if i in processed_indices:
            continue
        
        prev_close = data['close'].iloc[i - 1]
        curr_close = data['close'].iloc[i]
        curr_high = data['high'].iloc[i]
        curr_low = data['low'].iloc[i]
        ema = data[ema_col].iloc[i]
        atr = data[atr_col].iloc[i]
        date = data['date'].iloc[i]
        
        if pd.isna(ema) or pd.isna(atr):
            continue
        
        # ✅ УСЛО��ИЕ 1: Предыдущая свеча ВЫШЕ EMA
        prev_above_ema = prev_close > ema
        
        # ✅ УСЛОВИЕ 2: Текущая свеча в зоне касания ±0.15% от EMA
        tolerance = ema * 0.0015  # 0.15%
        touch_zone_lower = ema - tolerance
        touch_zone_upper = ema + tolerance
        curr_in_touch_zone = (curr_low <= touch_zone_upper and curr_high >= touch_zone_lower)
        
        if not (prev_above_ema and curr_in_touch_zone):
            continue
        
        # ✅ Касание найдено! Отслеживаем ГРУППУ свечей и определяем результат
        consolidation_indices = [i]
        consolidation_end_idx = i
        result = 'neutral'  # По умолчанию нейтральное
        
        max_consolidation = min(lookahead + 5, len(data) - i - 1)
        
        for j in range(i + 1, min(i + max_consolidation + 1, len(data))):
            future_close = data['close'].iloc[j]
            future_low = data['low'].iloc[j]
            future_high = data['high'].iloc[j]
            future_ema = data[ema_col].iloc[j]
            future_atr = data[atr_col].iloc[j]
            
            if pd.isna(future_ema) or pd.isna(future_atr):
                break
            
            future_lower = future_ema - future_atr
            future_upper = future_ema + future_atr
            future_tolerance = future_ema * 0.0015
            
            future_touch_lower = future_ema - future_tolerance
            future_touch_upper = future_ema + future_tolerance
            
            # ❌ ОТРИЦАТЕЛЬНОЕ: Закрытие ниже EMA-1ATR (полное пробитие)
            if future_close < future_lower:
                result = 'negative'
                consolidation_end_idx = j
                consolidation_indices.append(j)
                break
            
            # ✅ ПОЛОЖИТЕЛЬНОЕ: 
            # Цена пробила вниз (ниже EMA), но выше EMA-1ATR,
            # И потом пошла выше EMA (подтверждение разворота)
            if (future_close > future_lower and  # Не пробила полностью вниз
                future_close > future_ema):      # И пошла выше EMA (разворот)
                result = 'positive'
                consolidation_end_idx = j
                consolidation_indices.append(j)
                break
            
            # Цена ещё в зоне консолидации?
            still_in_zone = (future_low <= future_touch_upper and 
                            future_high >= future_touch_lower)
            
            if still_in_zone:
                # Продолжаем консолидацию (пока нет решения)
                consolidation_end_idx = j
                consolidation_indices.append(j)
            else:
                # Вышла из зоны без явного разворота
                consolidation_end_idx = j
                consolidation_indices.append(j)
                break
        
        # ✅ Добавляем ОДНО событие для ВСЕЙ группы свечей
        touches.append({
            'index': i,
            'date': date,
            'price': curr_close,
            'ema': ema,
            'atr': atr,
            'result': result
        })
        
        # ✅ Помечаем ВСЮ ГРУППУ как обработанную (чтобы не считать снова)
        for idx in consolidation_indices:
            processed_indices.add(idx)
    
    return touches

def create_touch_visualization(data, ticker, period, timeframe, touches):
    """Создание интерактивного графика"""
    fig = go.Figure()
    
    # Цена закрытия
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['close'],
        name='Close Price',
        line=dict(color='black', width=1),
        opacity=0.7
    ))
    
    # EMA
    fig.add_trace(go.Scatter(
        x=data['date'], y=data[f'ema{period}'],
        name=f'EMA{period}',
        line=dict(color='blue', width=2)
    ))
    
    # Верхняя граница (EMA + 1 ATR)
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data[f'ema{period}'] + data['atr'],
        name='EMA + 1 ATR',
        line=dict(dash='dash', color='green', width=1),
        showlegend=True
    ))
    
    # Нижняя граница (EMA - 1 ATR)
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data[f'ema{period}'] - data['atr'],
        name='EMA - 1 ATR',
        line=dict(dash='dash', color='red', width=1),
        fill='tonexty'
    ))
    
    # Распределяем касания по результатам
    touches_positive = [t for t in touches if t['result'] == 'positive']
    touches_negative = [t for t in touches if t['result'] == 'negative']
    touches_neutral = [t for t in touches if t['result'] == 'neutral']
    
    # ✅ ПОЛОЖИТЕЛЬНЫЕ касания
    if touches_positive:
        positive_dates = [t['date'] for t in touches_positive]
        positive_prices = [t['price'] for t in touches_positive]
        fig.add_trace(go.Scatter(
            x=positive_dates, y=positive_prices,
            mode='markers',
            name=f'Positive ✅ ({len(touches_positive)})',
            marker=dict(size=14, color='green', symbol='circle',
                       line=dict(color='darkgreen', width=2))
        ))
    
    # ❌ ОТРИЦАТЕЛЬНЫЕ касания
    if touches_negative:
        negative_dates = [t['date'] for t in touches_negative]
        negative_prices = [t['price'] for t in touches_negative]
        fig.add_trace(go.Scatter(
            x=negative_dates, y=negative_prices,
            mode='markers',
            name=f'Negative ❌ ({len(touches_negative)})',
            marker=dict(size=14, color='red', symbol='x',
                       line=dict(color='darkred', width=2))
        ))
    
    # ⚠️ НЕЙТРАЛЬНЫЕ касания
    if touches_neutral:
        neutral_dates = [t['date'] for t in touches_neutral]
        neutral_prices = [t['price'] for t in touches_neutral]
        fig.add_trace(go.Scatter(
            x=neutral_dates, y=neutral_prices,
            mode='markers',
            name=f'Neutral ⚠️ ({len(touches_neutral)})',
            marker=dict(size=12, color='orange', symbol='diamond')
        ))
    
    timeframe_label = "Daily (1D)" if timeframe == '1d' else "Weekly (1W)"
    
    fig.update_layout(
        title=f'{ticker} - EMA{period} ({timeframe_label}) - Touch Events',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=700,
        template='plotly_white'
    )
    
    return fig

# ============================================================
# ОСНОВНОЙ АНАЛИЗ
# ============================================================

for ticker in tickers:
    print(f"\n{'='*70}")
    print(f"📊 Анализирую {ticker}...")
    print(f"{'='*70}")
    
    try:
        for timeframe in timeframes:
            timeframe_label = "ДНЕВНОЙ" if timeframe == '1d' else "НЕДЕЛЬНЫЙ"
            print(f"\n  ⏱️  {timeframe_label} таймфрейм ({timeframe}):")
            
            try:
                # Загружаем данные
                data = yf.download(ticker, period="10y", interval=timeframe, progress=False)
                
                if data.empty:
                    print(f"    ⚠️  Данных нет")
                    continue
                
                data = data.reset_index()
                
                # Обработка MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                data.columns = data.columns.str.lower()
                
                # Проверяем необходимые колонки
                required_cols = ['date', 'high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    print(f"    ⚠️  Отсутствуют колонки: {missing_cols}")
                    continue
                
                # Преобразуем цены в float
                for col in ['open', 'high', 'low', 'close']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                data = data.dropna(subset=['close', 'high', 'low'])
                
                if data.empty:
                    print(f"    ⚠️  Нет валидных данных")
                    continue
                
                # Расчет EMA и ATR
                for period in ema_periods:
                    data[f'ema{period}'] = data['close'].ewm(span=period, adjust=False).mean()
                
                data = calculate_atr(data, atr_period)
                data = data.dropna(subset=['atr'])
                
                if data.empty:
                    print(f"    ⚠️  Нет данных после расчета ATR")
                    continue
                
                # ✅ АНАЛИЗ для каждого EMA периода
                for period in ema_periods:
                    touches = find_touch_events(data, f'ema{period}', 'atr', lookahead_days)
                    
                    if len(touches) >= min_total_interactions:
                        total_touches = len(touches)
                        positive_touches = len([t for t in touches if t['result'] == 'positive'])
                        negative_touches = len([t for t in touches if t['result'] == 'negative'])
                        neutral_touches = len([t for t in touches if t['result'] == 'neutral'])
                        
                        prob_positive = round((positive_touches / total_touches) * 100, 2)
                        prob_negative = round((negative_touches / total_touches) * 100, 2)
                        prob_neutral = round((neutral_touches / total_touches) * 100, 2)
                        
                        all_results[timeframe][f'EMA{period}'].append({
                            'Ticker': ticker,
                            'Total': total_touches,
                            'Positive': positive_touches,
                            'Negative': negative_touches,
                            'Neutral': neutral_touches,
                            'Prob_Pos (%)': prob_positive,
                            'Prob_Neg (%)': prob_negative,
                            'Prob_Neut (%)': prob_neutral
                        })
                        
                        print(f"    ✅ EMA{period}: {total_touches} | ✅{positive_touches}({prob_positive}%) | ❌{negative_touches}({prob_negative}%)")
                        
                        # Графики
                        try:
                            chart_dir = f'charts/{timeframe}'
                            if not os.path.exists(chart_dir):
                                os.makedirs(chart_dir)
                            fig = create_touch_visualization(data, ticker, period, timeframe, touches)
                            fig.write_html(f'{chart_dir}/{ticker}_EMA{period}.html')
                        except Exception as e:
                            print(f"      ⚠️ Ошибка графика: {e}")
            
            except Exception as e:
                print(f"    ❌ Ошибка: {str(e)}")
        
        print(f"\n  ✅ {ticker} готов")
    
    except Exception as e:
        print(f"  ❌ Ошибка: {str(e)}")

# ============================================================
# СОХРАНЕНИЕ В EXCEL
# ============================================================

print(f"\n{'='*70}")
print("💾 Сохраняю результаты...")
print(f"{'='*70}")

has_results = any(
    all_results[tf][f'EMA{p}'] 
    for tf in timeframes 
    for p in ema_periods
)

if not has_results:
    print("\n❌ Нет результатов!")
else:
    with pd.ExcelWriter('ema_touch_analysis_daily_weekly.xlsx', engine='openpyxl') as writer:
        for timeframe in timeframes:
            tf_label = "Daily" if timeframe == '1d' else "Weekly"
            
            for period in ema_periods:
                df = pd.DataFrame(all_results[timeframe][f'EMA{period}'])
                if not df.empty:
                    df = df.sort_values('Prob_Pos (%)', ascending=False)
                    sheet_name = f'{tf_label}_EMA{period}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✅ {sheet_name}: {len(df)} тикеров")
    
    print(f"\n✨ Готово! Файл 'ema_touch_analysis_daily_weekly.xlsx' создан")

print(f"\n{'='*70}")
print("📊 ИТОГОВАЯ ИНФОРМАЦИЯ:")
print(f"{'='*70}")
print(f"✅ EMA периоды: {', '.join(map(str, ema_periods))}")
print(f"✅ ATR период: {atr_period}")
print(f"✅ Lookahead: {lookahead_days} дней")
print(f"✅ Таймфреймы: Дневной (1D) и Недельный (1W)")
print(f"\n📌 ЛОГИКА КАСАНИЙ:")
print(f"  ✅ КАСАНИЕ: Цена подходит к EMA (±0.15%) из верхней зоны")
print(f"  ✅ ГРУППА: Несколько свечей подряд в зоне = 1 взаимодействие")
print(f"  ✅ ПОЛОЖИТЕЛЬНОЕ: Пробила вниз, но выше EMA-1ATR + потом выше EMA")
print(f"  ❌ ОТРИЦАТЕЛЬНОЕ: Закрытие ниже EMA - 1 ATR (полное пробитие)")
print(f"  ⚠️  НЕЙТРАЛЬНОЕ: Консолидация без явного разворота")
print(f"\n📈 ГРАФИКИ: charts/1d/ и charts/1wk/")
print(f"{'='*70}\n")