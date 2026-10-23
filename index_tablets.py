import os
import sqlite3

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
# Ajustez ce chemin si vos photos sont dans un sous-dossier spécifique après l'extraction
PHOTOS_DIR = "/workspace/CUNEISCANIA" 

print("Connexion à la base de données...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Création de la table pour les photos de tablettes
cursor.execute("""
CREATE TABLE IF NOT EXISTS tablet_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    filepath TEXT,
    status TEXT DEFAULT 'PENDING'
)
""")
conn.commit()

print(f"Indexation des photos de tablettes depuis : {PHOTOS_DIR}")
count = 0

# Extensions d'images courantes
valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

for root, dirs, files in os.walk(PHOTOS_DIR):
    # Éviter d'indexer les images de signes déjà stockées dans extracted_signs
    if "extracted_signs" in root:
        continue
        
    for file in files:
        if file.lower().endswith(valid_extensions):
            filepath = os.path.join(root, file)
            
            # Vérifier si la photo n'est pas déjà dans la BDD
            cursor.execute("SELECT id FROM tablet_photos WHERE filepath = ?", (filepath,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO tablet_photos (filename, filepath, status)
                    VALUES (?, ?, 'PENDING')
                """, (file, filepath))
                count += 1

conn.commit()
conn.close()

print(f"Succès ! {count} nouvelles photos de tablettes ont été indexées dans la base de données.")
