import os
import sqlite3
import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"

# Chargement du modèle au démarrage (ou à la demande)
def load_ai_model():
    print("\n[IA] Chargement du modèle de traduction...")
    path = kagglehub.model_download("max1234567a/cuneiformbase-400m-akkadian-fn-v3-stage3/transformers/default")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    print(f"[IA] Modèle chargé avec succès sur {device.upper()} !")
    return tokenizer, model, device

def menu_translate(tokenizer, model, device):
    print("\n--- 🔤 TRADUCTION / TRANSLITTÉRATION AKKADIENNE ---")
    text = input("Entrez le texte translittéré (ex: a-na am-i-li) : ")
    if not text:
        return
    
    prompt = f"Translate Akkadian to English: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nRésultat : {result}\n")
    
    # Enregistrement optionnel dans la BDD
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO translations_log (input_text, translated_text) VALUES (?, ?)", (text, result))
    conn.commit()
    conn.close()

def menu_search_sign():
    print("\n--- 🔍 RECHERCHE DE SIGNE DANS LA BDD ---")
    query = input("Entrez le nom ou une partie du signe recherché (ex: AB, A2) : ")
    if not query:
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, filepath FROM sign_images WHERE filename LIKE ? LIMIT 10", (f"%{query}%",))
    rows = cursor.fetchall()
    conn.close()
    
    if rows:
        print(f"\nTrouvé {len(rows)} résultats (affichage des 10 premiers) :")
        for filename, filepath in rows:
            print(f" - {filename} -> {filepath}")
    else:
        print("\nAucun signe trouvé pour cette recherche.")

def main():
    # Initialisation de l'IA
    tokenizer, model, device = load_ai_model()
    
    while True:
        print("\n================================")
        print("    CUNEISCANIA - MENU V1       ")
        print("================================")
        print("1. Traduire un texte akkadien")
        print("2. Rechercher un signe visuel (BDD)")
        print("3. Quitter")
        
        choice = input("\nVotre choix (1-3) : ")
        
        if choice == "1":
            menu_translate(tokenizer, model, device)
        elif choice == "2":
            menu_search_sign()
        elif choice == "3":
            print("\nFermeture de Cuneiscania. À bientôt !")
            break
        else:
            print("Choix invalide, veuillez entrer 1, 2 ou 3.")

if __name__ == "__main__":
    main()
