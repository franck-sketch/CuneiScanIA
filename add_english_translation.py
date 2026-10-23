import sqlite3

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"

# Dictionnaire de traduction sémantique (Sumérien/Akkadien -> Anglais)
LEXICON_MAP = {
    "diš": "Line",
    "še": "barley",
    "ensi2": "governor",
    "ba-ni": "allocated / built",
    "dub": "tablet",
    "a-na": "to / for",
    "lugal": "king",
    "gur": "gur (capacity unit)",
    "ki": "place / from"
}

def translate_to_english(translit_text):
    parts = translit_text.split()
    translated_parts = []
    for p in parts:
        if "-kam" in p:
            year_num = p.replace("-kam", "")
            translated_parts.append(f"year {year_num}")
        elif p in LEXICON_MAP:
            translated_parts.append(LEXICON_MAP[p])
        else:
            translated_parts.append(p)
    return " | ".join(translated_parts)

def upgrade_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ajouter la colonne english_text si elle n'existe pas
    cursor.execute("PRAGMA table_info(translations_log);")
    columns = [col[1] for col in cursor.fetchall()]
    if 'english_text' not in columns:
        cursor.execute("ALTER TABLE translations_log ADD COLUMN english_text TEXT;")
        conn.commit()
        print(" -> Colonne 'english_text' ajoutée à la table translations_log.")

    # Mettre à jour les entrées existantes
    cursor.execute("SELECT id, translated_text FROM translations_log")
    rows = cursor.fetchall()
    print(f" -> Génération des traductions en anglais pour {len(rows)} lignes...")

    for row_id, translit in rows:
        eng = translate_to_english(translit)
        cursor.execute("UPDATE translations_log SET english_text = ? WHERE id = ?", (eng, row_id))
    
    conn.commit()
    
    # Afficher un échantillon mis à jour
    cursor.execute("SELECT image_name, translated_text, english_text FROM translations_log ORDER BY RANDOM() LIMIT 3")
    samples = cursor.fetchall()
    print("\n[Vérification] Échantillon enrichi (Translittération & Traduction Anglaise) :\n" + "="*70)
    for img, trans, eng in samples:
        print(f" 🗂️ Tablette/Ligne : {img}")
        print(f" 📜 Translit.     : {trans}")
        print(f" 🇬🇧 Traduction Anglais : {eng}")
        print("-" * 70)
        
    conn.close()

if __name__ == "__main__":
    upgrade_logs()
