import os
import sqlite3

# Chemins de travail
db_path = "/workspace/CUNEISCANIA/cuneiscania.db"
images_dir = "/workspace/CUNEISCANIA/extracted_signs"  # Adaptez si le dossier d'extraction a un autre nom

print("Connexion à la base de données SQLite...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Création de la table pour les images de signes
cursor.execute("""
CREATE TABLE IF NOT EXISTS sign_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    filepath TEXT
)
""")
conn.commit()

print(f"Parcours du dossier : {images_dir}")
count = 0

# Parcours récursif du dossier pour trouver toutes les images .png
for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = file
            filepath = os.path.join(root, file)
            
            # Insertion dans la base de données
            cursor.execute("""
                INSERT INTO sign_images (filename, filepath)
                VALUES (?, ?)
            """, (filename, filepath))
            count += 1

conn.commit()
conn.close()

print(f"Succès ! {count} images de signes ont été indexées dans la base de données.")
