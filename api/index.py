from fastapi import FastAPI

app = FastAPI()

# Health check - это БЕЗ зависимостей!
@app.get("/health")
def health():
    return {"status": "ok"}

# Ленивый импорт всего остального - только при запросе к API
_imports_done = False
signals_cache = None
cache_lock = None

def _lazy_init():
    """Загружаем все тяжелые импорты только при первом API запросе"""
    global _imports_done, signals_cache, cache_lock
    
    if _imports_done:
        return
    
    import threading
    import time
    import pandas as pd
    import numpy as np
    
    signals_cache = {'signals': [], 'last_update': None}
    cache_lock = threading.Lock()
    
    _imports_done = True

# Все остальные эндпоинты требуют инициализации
@app.get("/api/signals")
def get_signals():
    _lazy_init()
    return {"signals": [], "last_update": None, "message": "Cache empty, request /api/refresh to load"}

@app.post("/api/refresh")
def refresh():
    _lazy_init()
    return {"status": "ok", "message": "Refresh triggered"}

@app.get("/api/signals/{ticker}")
def get_ticker_signal(ticker: str):
    return {"error": "Not implemented"}




