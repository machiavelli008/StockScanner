import sys
from pathlib import Path

# Добавляем корень проекта в path для импорта backend
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импортируем FastAPI app из backend
from backend.main import app

# Vercel/Railway автоматически использует эту переменную 'app' как entrypoint






