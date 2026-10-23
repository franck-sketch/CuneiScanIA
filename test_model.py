import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from collections import Counter

MODEL_PATH = "/workspace/CUNEISCANIA/cunei_ocr_model.pth"
CLASSES_PATH = "/workspace/CUNEISCANIA/cunei_classes.json"
CSV_PATH = "/workspace/CUNEISCANIA/ocr_dataset_clustered.csv"

# Chargement dynamique des classes
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        ACCADIAN_CLASSES = json.load(f)
else:
    raise FileNotFoundError(f"Le fichier {CLASSES_PATH} est introuvable.")

def test_inference_50():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Inference] Chargement du modèle depuis {MODEL_PATH}...")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(ACCADIAN_CLASSES))
    
    if not os.path.exists(MODEL_PATH):
        print("[Erreur] Le fichier de modèle est introuvable. Lance d'abord train_ocr.py.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        valid_df = df[df['cluster_id'] != -1]
        
        n_test = min(50, len(valid_df))
        test_samples = valid_df.sample(n_test)
        
        predictions_counter = Counter()
        print(f"\n[Inference] Test d'évaluation en cours sur {n_test} photos aléatoires...")
        
        with torch.no_grad():
            for _, row in test_samples.iterrows():
                img_path = row['filepath']
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted_idx = torch.max(probabilities, 1)
                    
                    predicted_sign = ACCADIAN_CLASSES[predicted_idx.item()]
                    predictions_counter[predicted_sign] += 1
                except Exception:
                    pass
        
        print("\n--- Bilan de la répartition des prédictions sur les 50 photos ---")
        for sign, count in predictions_counter.most_common():
            print(f" -> Signe '{sign}': {count} fois sur {n_test} ({(count/n_test)*100:.1f}%)")
    else:
        print(f"[Erreur] Le fichier CSV {CSV_PATH} est introuvable.")

if __name__ == "__main__":
    test_inference_50()
