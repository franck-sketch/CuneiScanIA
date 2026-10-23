import os
import glob
import warnings
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Désactiver les avertissements liés au certificat SSL non vérifié
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. Détection automatique du fichier CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(script_dir, 'collections*.csv'))

if not csv_files:
    csv_files = glob.glob('collections*.csv')

if not csv_files:
    raise FileNotFoundError("Aucun fichier CSV commençant par 'collections' n'a été trouvé.")

csv_file = csv_files[0]
print(f"Fichier CSV détecté avec succès : {csv_file}")

# Charger le fichier CSV
df = pd.read_csv(csv_file)

# 2. Créer l'arborescence pour le dataset
output_dir = 'dataset_babylonien'
images_dir = os.path.join(output_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

# 3. Configurer une session HTTP robuste
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# 4. Choisir la quantité : 2500 pour un échantillon, ou None pour tout télécharger
LIMIT = 2500  
subset = df.head(LIMIT) if LIMIT else df

downloaded_records = []

print(f"Début du téléchargement de {len(subset)} images vers le dossier '{images_dir}'...")

for idx, row in subset.iterrows():
    img_url = row.get('Image')
    if pd.isna(img_url):
        continue

    file_name = f"tablet_{idx}.jpg"
    file_path = os.path.join(images_dir, file_name)

    try:
        # Ajout de verify=False pour contourner l'erreur de certificat SSL du serveur distant
        response = session.get(img_url, timeout=10, verify=False)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            record = row.to_dict()
            record['local_image_path'] = file_path
            downloaded_records.append(record)

            if (idx + 1) % 100 == 0:
                print(f"Progression : {idx + 1}/{len(subset)} images téléchargées.")
        else:
            print(f"Erreur HTTP {response.status_code} pour l'URL : {img_url}")
    except Exception as e:
        print(f"Erreur réseau pour {img_url}: {e}")

# 5. Sauvegarder le fichier de métadonnées
if downloaded_records:
    df_downloaded = pd.DataFrame(downloaded_records)
    meta_path = os.path.join(output_dir, 'metadata_dataset.csv')
    df_downloaded.to_csv(meta_path, index=False)
    print(f"\nSuccès ! {len(downloaded_records)} images ont été téléchargées dans '{images_dir}'.")
    print(f"Le fichier de métadonnées a été sauvegardé ici : '{meta_path}'")
else:
    print("Aucune image n'a pu être téléchargée.")
