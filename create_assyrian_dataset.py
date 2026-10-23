import sqlite3
import pandas as pd
import os
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from pathlib import Path

# 1. Connexion à la base de données
db_path = 'cuneiscania.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 2. Création de la table assyrian_tablet_photos si elle n'existe pas
cursor.execute("""
CREATE TABLE IF NOT EXISTS assyrian_tablet_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT,
    museum_number TEXT,
    image_url TEXT,
    filepath TEXT,
    status TEXT DEFAULT 'PENDING',
    culture TEXT
)
""")
conn.commit()

# On corrige les éventuelles erreurs de statut
cursor.execute("UPDATE assyrian_tablet_photos SET status = 'PENDING' WHERE status != 'DOWNLOADED'")
conn.commit()

# 3. Insertion des données si la table est vide
cursor.execute("SELECT COUNT(*) FROM assyrian_tablet_photos")
count = cursor.fetchone()[0]

if count == 0:
    files = [
        'collections-26-10-22-20_03_40.csv',
        'collections-26-10-22-20_02_38.csv'
    ]
    inserted_count = 0
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                img_url = str(row.get('Image', ''))
                museum_num = str(row.get('Museum number', ''))
                culture = str(row.get('Culture', ''))
                if img_url and img_url != 'nan':
                    filename = img_url.split('/')[-1]
                    filepath = f"/workspace/CUNEISCANIA/assyrian_images/{filename}"
                    cursor.execute("""
                        INSERT INTO assyrian_tablet_photos (source_file, museum_number, image_url, filepath, status, culture)
                        VALUES (?, ?, ?, ?, 'PENDING', ?)
                    """, (f, museum_num, img_url, filepath, culture))
                    inserted_count += 1
    conn.commit()
    print(f"Insertion de {inserted_count} enregistrements dans assyrian_tablet_photos.")
else:
    print(f"La table contient déjà {count} enregistrements.")

# 4. Téléchargement des images en attente
output_dir = Path("/workspace/CUNEISCANIA/assyrian_images")
output_dir.mkdir(parents=True, exist_ok=True)

cursor.execute("SELECT id, image_url, filepath FROM assyrian_tablet_photos WHERE status = 'PENDING'")
records = cursor.fetchall()

print(f"Début du téléchargement pour {len(records)} images...")

for rec_id, image_url, filepath in records:
    try:
        if not image_url or image_url == 'nan':
            continue
            
        response = requests.get(image_url, timeout=15, verify=False)
        if response.status_code == 200:
            filename = image_url.split('/')[-1]
            local_path = output_dir / filename
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            cursor.execute("""
                UPDATE assyrian_tablet_photos 
                SET status = 'DOWNLOADED', filepath = ? 
                WHERE id = ?
            """, (str(local_path), rec_id))
            conn.commit()
        else:
            print(f"Erreur HTTP {response.status_code} pour l'URL : {image_url}")
    except Exception as e:
        print(f"Erreur lors du téléchargement de {image_url}: {e}")

print("Traitement du corpus assyrien terminé avec succès !")
conn.close()
