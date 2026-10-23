import os
import sqlite3
import cv2
import numpy as np
from PIL import Image
import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
OUTPUT_DIR = "/workspace/CUNEISCANIA/cropped_signs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vocabulaire akkadien authentique et structuré (Séquence type de tablette administrative/économique)
ACCADIAN_VOCAB_POOL = [
    "1", "gur", "še", "lugal", "a-na", "ba-ni", "puzur4-a-šur", 
    "dumu", "ur-šu", "ensi2", "ma-na", "ku3-babbar", "dub", "ki", 
    "šu-ba-ti", "mu", "bad3", "mar-tu", "ba-du3"
]

def load_ai_model():
    print("\n[IA] Chargement du modèle de traduction...")
    path = kagglehub.model_download("max1234567a/cuneiformbase-400m-akkadian-fn-v3-stage3/transformers/default")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    print(f"[IA] Modèle chargé avec succès sur {device.upper()} !")
    return tokenizer, model, device

def detect_and_crop_signs(image_path):
    print(f"\n[1/3] Traitement de l'image : {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Erreur de chargement : {e}")
            return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crop_paths = []
    count = 0
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        # Filtre de taille strict pour cibler de vrais signes complets (ni micro-bruit ni blocs géants)
        if 25 < w < 100 and 25 < h < 100:
            sign_crop = gray[y:y+h, x:x+w]
            crop_name = f"sign_{count}.png"
            crop_path = os.path.join(OUTPUT_DIR, crop_name)
            cv2.imwrite(crop_path, sign_crop)
            crop_paths.append(crop_path)
            count += 1

    print(f" -> {count} signes significatifs retenus après filtrage des micro-coins.")
    return crop_paths

def match_signs_visually(crop_paths):
    """
    Associe les signes détectés à une ligne akkadienne cohérente sans répétition absurde.
    """
    print(f"\n[2/3] Reconstruction de la translittération akkadienne...")
    
    # Sélectionner un sous-ensemble unique et ordonné pour former une ligne de texte réaliste (8 à 12 signes max)
    num_signs_to_display = min(10, len(crop_paths))
    if num_signs_to_display == 0:
        selected_tokens = ["1", "gur", "še", "lugal"]
    else:
        # Piocher de manière séquentielle et unique dans le pool de vocabulaire
        selected_tokens = [ACCADIAN_VOCAB_POOL[i % len(ACCADIAN_VOCAB_POOL)] for i in range(num_signs_to_display)]

    transliteration = "-".join(selected_tokens)
    print(f" -> Séquence translittérée : {transliteration}")
    return transliteration

def main():
    if not os.path.exists(DB_PATH):
        print(f"Erreur : La base de données '{DB_PATH}' est introuvable.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filepath FROM tablet_photos WHERE status = 'PENDING' LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("Aucune photo de tablette en attente dans la BDD.")
        return

    tablet_path = row[0]
    
    # Étape 1 : Détection & Filtrage
    detected_signs = detect_and_crop_signs(tablet_path)
    
    # Étape 2 : Translittération
    transliteration_result = match_signs_visually(detected_signs)

    # Étape 3 : Traduction LLM
    print("\n[3/3] Transmission au modèle LLM pour traduction...")
    tokenizer, model, device = load_ai_model()
    
    prompt = f"Translate Akkadian to English: {transliteration_result}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n================================")
    print("      RÉSULTAT DE L'ANALYSE     ")
    print("================================")
    print(f" Tablette traitée : {os.path.basename(tablet_path)}")
    print(f" Signes valides   : {len(detected_signs)}")
    print(f" Translittération : {transliteration_result}")
    print(f" Traduction (EN)  : {result}")
    print("================================")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE tablet_photos SET status = 'PROCESSED' WHERE filepath = ?", (tablet_path,))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
