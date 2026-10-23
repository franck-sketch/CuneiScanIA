import sqlite3
import pandas as pd
import os

db_path = "/workspace/CUNEISCANIA/cuneiscania.db"
print("Création de la base de données SQLite...")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Création des tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS old_akkadian_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS old_akkadian_signs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sign TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ur3_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ur3_signs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sign_key TEXT,
    sign_value TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS translations_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    input_text TEXT,
    translated_text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
print("Tables créées avec succès.")

# 2. Importation de old-akkadian-words.csv
if os.path.exists("old-akkadian-words.csv"):
    df = pd.read_csv("old-akkadian-words.csv", header=None, names=["word"])
    df.to_sql("old_akkadian_words", conn, if_exists="replace", index=False)
    print(f"-> {len(df)} mots d'Akkadien ancien insérés.")

# 3. Importation de old-akkadian-signs.csv
if os.path.exists("old-akkadian-signs.csv"):
    df = pd.read_csv("old-akkadian-signs.csv", header=None, names=["sign"])
    df.to_sql("old_akkadian_signs", conn, if_exists="replace", index=False)
    print(f"-> {len(df)} signes d'Akkadien ancien insérés.")

# 4. Importation de ur3-words.csv
if os.path.exists("ur3-words.csv"):
    df = pd.read_csv("ur3-words.csv", header=None, names=["word"])
    df.to_sql("ur3_words", conn, if_exists="replace", index=False)
    print(f"-> {len(df)} mots Ur III insérés.")

# 5. Importation de ur3-signs.csv (avec gestion des erreurs de colonnes)
if os.path.exists("ur3-signs.csv"):
    try:
        df = pd.read_csv("ur3-signs.csv", header=None, names=["sign_key", "sign_value"], on_bad_lines='skip')
        df.to_sql("ur3_signs", conn, if_exists="replace", index=False)
        print(f"-> {len(df)} signes Ur III insérés.")
    except Exception as e:
        print(f"Erreur lors de l'import de ur3-signs.csv : {e}")

conn.close()
print("\nBase de données prête et sauvegardée sous :", db_path)
