import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

MODEL_SAVE_PATH = "/workspace/CUNEISCANIA/cunei_ocr_model.pth"
CLASSES_PATH = "/workspace/CUNEISCANIA/cunei_classes.json"
CSV_PATH = "/workspace/CUNEISCANIA/ocr_dataset_clustered.csv"

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        ACCADIAN_CLASSES = json.load(f)
else:
    raise FileNotFoundError(f"Le fichier {CLASSES_PATH} est introuvable.")

class CuneiDataset(Dataset):
    def __init__(self, df_cleaned, transform=None):
        self.df = df_cleaned.copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label = row['label_idx']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (64, 64), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model():
    print(f"\n[OCR Training] Nettoyage et équilibrage strict (max 100 imgs/classe)...")
    
    if not os.path.exists(CSV_PATH):
        print(f"[Erreur] Le fichier {CSV_PATH} est introuvable.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. Exclure le bruit strict (-1)
    df = df[df['cluster_id'] != -1].copy()

    # 2. Exclure le méga-cluster poubelle (le plus grand)
    cluster_counts = df['cluster_id'].value_counts()
    garbage_cluster = cluster_counts.index[0]
    print(f" -> Exclusion du cluster poubelle : ID {garbage_cluster} ({cluster_counts.iloc[0]} images)")
    df = df[df['cluster_id'] != garbage_cluster].copy()

    # 3. Sélectionner exactement 14 clusters stables
    remaining_clusters = df['cluster_id'].value_counts().index
    selected_clusters = remaining_clusters[:len(ACCADIAN_CLASSES)]
    df = df[df['cluster_id'].isin(selected_clusters)].copy()

    # 4. Plafonnement strict à 100 images par cluster pour un équilibre parfait
    max_samples_per_cluster = 100 
    balanced_dfs = []
    for cid in selected_clusters:
        group = df[df['cluster_id'] == cid]
        if len(group) > max_samples_per_cluster:
            group = group.sample(max_samples_per_cluster, random_state=42)
        balanced_dfs.append(group)
    
    df = pd.concat(balanced_dfs).reset_index(drop=True)

    # 5. Mapping 1 à 1 strict
    cluster_to_class = {cid: i for i, cid in enumerate(selected_clusters)}
    df['label_idx'] = df['cluster_id'].map(cluster_to_class)

    print(f" -> Nombre total d'exemples parfaitement équilibrés : {len(df)}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CuneiDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OCR Training] Entraînement sur l'appareil : {str(device).upper()}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(ACCADIAN_CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"\n[OCR Training] Début de l'apprentissage équilibré ({epochs} époques)...")
    
    best_loss = float('inf')
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f" -> Époque [{epoch+1}/{epochs}] - Perte: {epoch_loss:.4f} - Précision: {epoch_acc:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n[Succès] Modèle équilibré sauvegardé sous : {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
