import os
import sqlite3

db_path = os.path.join(os.path.dirname(__file__), '../../.agents/memory/memoria.db')
conn = sqlite3.connect(os.path.abspath(db_path))
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
tables = [table[0] for table in tables]
print("Tablas encontradas:", tables)

for table in tables:
    print(f"\n--- Tabla: {table} ---")
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    columns = [col[2] for col in columns]
    print(f"Columnas: {columns}")


    cursor.execute(f"SELECT * FROM {table} LIMIT 20;")
    rows = cursor.fetchall()
    print("Filas (máx 20):\n")
    for row in rows:
        print(row)

conn.close()
