import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import DBSCAN
import numpy as np
import shutil

CSV_PATH = "/workspace/CUNEISCANIA/ocr_dataset.csv"
OUTPUT_CSV = "/workspace/CUNEISCANIA/ocr_dataset_clustered.csv"
CLUSTER_DIR = "/workspace/CUNEISCANIA/clusters"

# ==========================================
# PARAMÈTRES DE DISTANCE AJUSTABLES (DBSCAN)
# ==========================================
# - DBSCAN_EPS : Seuil de distance cosinus. 
#   * Plus la valeur est petite (ex: 0.20), plus le regroupement est strict (petits clusters précis).
#   * Plus la valeur est grande (ex: 0.50), plus le regroupement tolère des variations (grands clusters).
# - DBSCAN_MIN_SAMPLES : Nombre min d'exemplaires requis pour valider un groupe de signes.
DBSCAN_EPS = 0.23
DBSCAN_MIN_SAMPLES = 2

class CuneiInferenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (64, 64), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, row['filename']

def run_clustering():
    print("\n[Clustering] Chargement du modèle de vision (ResNet18) pour extraction d'embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ResNet18 sans la couche finale de classification (vecteurs de 512 dimensions)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CuneiInferenceDataset(CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    embeddings_list = []
    filenames_list = []

    print("[Clustering] Extraction des caractéristiques visuelles sur les crops...")
    with torch.no_grad():
        for inputs, fnames in dataloader:
            inputs = inputs.to(device)
            outputs = resnet(inputs)
            embeddings_list.append(outputs.cpu().numpy())
            filenames_list.extend(fnames)

    X = np.vstack(embeddings_list)
    print(f"[Clustering] Forme de la matrice d'embeddings : {X.shape}")

    # Application de DBSCAN (contrôle par distance et gestion du bruit)
    print(f"[Clustering] Regroupement par densité (DBSCAN: eps={DBSCAN_EPS}, metric='cosine')...")
    dbscan = DBSCAN(eps=DBSCAN_EPS, metric="cosine", min_samples=DBSCAN_MIN_SAMPLES)
    cluster_labels = dbscan.fit_predict(X)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"[Clustering] Nombre de clusters valides trouvés : {n_clusters}")
    print(f"[Clustering] Nombre d'éléments isolés / bruit (étiquetés -1) : {n_noise}")

    # Sauvegarde des résultats dans un nouveau CSV enrichi
    df = pd.read_csv(CSV_PATH)
    df['cluster_id'] = cluster_labels
    df.to_csv(OUTPUT_CSV, index=False)

    # Nettoyage et organisation d'exemples par cluster dans des dossiers
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    # Copier les 10 premiers exemples de chaque cluster (et le bruit à part)
    for cid in set(cluster_labels):
        if cid == -1:
            c_path = os.path.join(CLUSTER_DIR, "noise_outliers")
        else:
            c_path = os.path.join(CLUSTER_DIR, f"cluster_{cid}")
        
        os.makedirs(c_path, exist_ok=True)
        sample_rows = df[df['cluster_id'] == cid].head(10)
        for _, row in sample_rows.iterrows():
            src = row['filepath']
            dst = os.path.join(c_path, row['filename'])
            if os.path.exists(src):
                try:
                    shutil.copy(src, dst)
                except Exception:
                    pass

    print(f"\n[Succès] Clustering et nettoyage terminés avec succès !")
    print(f" - CSV mis à jour : {OUTPUT_CSV}")
    print(f" - Dossier d'échantillons visuels : {CLUSTER_DIR}/")

if __name__ == "__main__":
    run_clustering()
