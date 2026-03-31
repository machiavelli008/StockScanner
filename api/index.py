import os
import sys
from pathlib import Path

# Убедимся, что backend модуль импортируется правильно
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# На Vercel импортируем только функции и переменные, БЕЗ инициализации пути для статики
# чтобы избежать ошибок при startup
try:
    from backend.main import app as backend_app
    # FastAPI app готов к использованию
    app = backend_app
except Exception as e:
    print(f"Error importing backend: {e}")
    # Fallback - если импорт падает, создаём минимальный app
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/api/health")
    def health():
        return {"status": "error", "message": str(e)}

