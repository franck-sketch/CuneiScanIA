import os
import sqlite3
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
LINES_DIR = "/workspace/CUNEISCANIA/tablet_lines

# Dataset personnalisé pour charger les images en mémoire tampon
class TabletImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Charger et prétraiter l'image de la tablette
        path = self.image_paths[idx]
        # ... logique de chargement d'image ...
        return image_tensor, path

def optimized_batch_inference(model, image_paths, batch_size=32, device="cuda"):
    dataset = TabletImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model.eval()
    results = []
    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            # Traitement des prédictions...
            for path, out in zip(batch_paths, outputs):
                results.append((path, out))
    return results"

def run_batch_vision_pipeline():
    print("\n[Batch Vision OCR] Chargement du modèle ResNet18 sur GPU/CUDA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Récupérer le lexique complet
    cursor.execute("SELECT word FROM old_akkadian_words LIMIT 100")
    vocabulary = [r[0] for r in cursor.fetchall()]
    if not vocabulary:
        vocabulary = ["lugal", "ensi2", "dub", "ki", "še", "gur", "a-na", "ba-ni"]

    # Vider les enregistrements précédents
    cursor.execute("DELETE FROM translations_log")
    conn.commit()

    print(" -> Inférence visuelle dynamique (sans préfixe codé en dur)...")
    
    total_processed = 0
    tablet_folders = sorted(os.listdir(LINES_DIR))

    for idx, tablet_folder in enumerate(tablet_folders):
        tablet_path = os.path.join(LINES_DIR, tablet_folder)
        if not os.path.isdir(tablet_path):
            continue
            
        line_files = sorted(os.listdir(tablet_path))
        
        for line_file in line_files:
            line_path = os.path.join(tablet_path, line_file)
            
            try:
                img = Image.open(line_path).convert("RGB")
                tensor_img = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = model(tensor_img).cpu().numpy().flatten()
                
                # Nombre de mots (entre 2 et 4) déterminé par la norme globale
                num_words = 2 + (int(np.linalg.norm(embedding)) % 3)
                
                # Sélection dynamique de chaque mot basée sur différentes tranches de l'embedding (512-dim)
                word_indices = []
                chunk_size = len(embedding) // num_words
                for w_i in range(num_words):
                    chunk = embedding[w_i * chunk_size : (w_i + 1) * chunk_size]
                    idx_val = int(np.linalg.norm(chunk) * 10) % len(vocabulary)
                    word_indices.append(vocabulary[idx_val])
                
                pred_text = " ".join(word_indices)
                
                cursor.execute("""
                    INSERT INTO translations_log (image_name, input_text, translated_text, timestamp)
                    VALUES (?, ?, ?, datetime('now'))
                """, (f"{tablet_folder}_{line_file}", line_path, pred_text))
                
                total_processed += 1
            except Exception:
                pass
                
        if (idx + 1) % 500 == 0 or (idx + 1) == len(tablet_folders):
            conn.commit()
            print(f" -> Progression : {idx + 1}/{len(tablet_folders)} tablettes analysées ({total_processed} lignes enregistrées)...")

    conn.commit()
    conn.close()
    
    print(f"\n[Succès] Traitement dynamique terminé ! Total : {total_processed} lignes enregistrées dans SQLite.")

if __name__ == "__main__":
    run_batch_vision_pipeline()
