import os
import sys
from pathlib import Path

# Убедимся, что backend модуль импортируется правильно
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импортируем FastAPI app, но без монтирования статики
from backend.main import (
    app, get_signals, refresh_signals_endpoint, 
    get_signal_by_ticker, startup_event
)

# На Vercel монтировать статику нельзя, только API
# Статика будет раздаваться из public/ папки через vercel.json
