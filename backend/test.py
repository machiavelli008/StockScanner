import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
index_path = os.path.join(FRONTEND_DIR, "index.html")

print(f"BASE_DIR: {BASE_DIR}")
print(f"FRONTEND_DIR: {FRONTEND_DIR}")
print(f"index_path: {index_path}")
print(f"Файл существует: {os.path.exists(index_path)}")