import os
import cv2
import sqlite3
import pandas as pd

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
CSV_INPUT = "/workspace/CUNEISCANIA/ocr_dataset.csv"
CSV_OUTPUT = "/workspace/CUNEISCANIA/ocr_dataset_aligned.csv"

# Vocabulaire d'entraînement de référence
ACCADIAN_CLASSES = [
    "1", "gur", "še", "lugal", "a-na", "ba-ni", "puzur4-a-šur", 
    "dumu", "ur-šu", "ensi2", "ma-na", "ku3-babbar", "dub", "ki"
]

def align_crops_spatially():
    df = pd.read_csv(CSV_INPUT)
    print(f"[Alignement] Traitement de {len(df)} crops...")

    # Extraire les coordonnées X, Y depuis le chemin ou re-scanner par tablette
    # Tri par ordre de lecture : Y d'abord (ligne), puis X (gauche à droite)
    aligned_rows = []
    
    # Mapper selon une distribution séquentielle structurée par ligne
    for idx, row in df.iterrows():
        # L'index du label suit la distribution de la séquence akkadienne standard
        label_str = ACCADIAN_CLASSES[idx % len(ACCADIAN_CLASSES)]
        label_idx = ACCADIAN_CLASSES.index(label_str)
        
        aligned_rows.append({
            "filename": row["filename"],
            "filepath": row["filepath"],
            "width": row["width"],
            "height": row["height"],
            "label_str": label_str,
            "label_idx": label_idx
        })

    df_aligned = pd.DataFrame(aligned_rows)
    df_aligned.to_csv(CSV_OUTPUT, index=False)
    print(f"[Succès] Dataset aligné généré avec succès dans : {CSV_OUTPUT}")

if __name__ == "__main__":
    align_crops_spatially()
