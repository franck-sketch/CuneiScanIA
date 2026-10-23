import os
import time
import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import SequenceMatcher

GROUND_TRUTH_BENCHMARK = {
    "P361098": {
        "parts": [
            {
                "label": "Formule d'en-tête (Adresse)",
                "transliteration_ref": "um-ma puzur4-a-šur-ma a-na bu-za-zu qi2-bi-ma",
                "translation_ref": "Thus says Puzur-Ashur: Speak to Buzazu..."
            },
            {
                "label": "Corps du texte (Transactions)",
                "transliteration_ref": "10 ma-na ku3-babbar a-na puzur4-a-šur a-na-ki-im ku3-babbar a-na a-šur-i-din id-di-in",
                "translation_ref": "10 minas of silver to Puzur-Ashur, as for the silver, he gave it to Ashur-iddin."
            }
        ]
    }
}

def load_ai_model():
    print("\n[IA] Chargement du modèle de traduction...")
    model_name = "max1234567a/cuneiformbase-400m-akkadian-fn-v3-stage3/transformers/default"
    
    path = None
    for attempt in range(1, 4):
        try:
            path = kagglehub.model_download(model_name)
            break
        except Exception:
            time.sleep(2)
            
    if not path:
        raise RuntimeError("Impossible de contacter KaggleHub.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    print(f"[IA] Modèle prêt sur {device.upper()} !")
    return tokenizer, model, device

def compute_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() * 100

def run_benchmark_evaluation(tablet_id):
    if tablet_id not in GROUND_TRUTH_BENCHMARK:
        print(f"Erreur : Aucune référence trouvée pour '{tablet_id}'.")
        return

    tablet_data = GROUND_TRUTH_BENCHMARK[tablet_id]
    print(f"\n==================================================")
    print(f" BENCHMARK QUANTITATIF (NORMALLISÉ) : {tablet_id}")
    print(f"==================================================")

    tokenizer, model, device = load_ai_model()
    total_score = 0

    for idx, part in enumerate(tablet_data["parts"], 1):
        print(f"\n--- Section {idx} : {part['label']} ---")
        print(f"Translittération : {part['transliteration_ref']}")
        print(f"Référence (Expert) : {part['translation_ref']}")

        prompt = f"Translate Akkadian to English: {part['transliteration_ref']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Génération équilibrée : Beam search modéré sans pénalité d'ngram stricte
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=3,
            length_penalty=1.0,
            early_stopping=True
        )
        ai_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        score = compute_similarity(ai_translation, part['translation_ref'])
        total_score += score

        print(f"Traduction (IA)    : {ai_translation}")
        print(f"-> Score de similarité textuelle : {score:.1f}%")
    
    avg_score = total_score / len(tablet_data["parts"])
    print(f"\n==================================================")
    print(f" SCORE GLOBAL DE FIDÉLITÉ : {avg_score:.1f}%")
    print(f"==================================================")

if __name__ == "__main__":
    run_benchmark_evaluation("P361098")
