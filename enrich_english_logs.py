import sqlite3

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"

# Dictionnaire de correspondance sémantique
LEXICON_MAP = {
    "lugal": "king",
    "ensi2": "governor",
    "dub": "tablet",
    "ki": "place / from",
    "še": "barley",
    "gur": "gur (capacity unit)",
    "a-na": "to / for",
    "ba-ni": "allocated",
    "diš": "line"
}

def enrich_english():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # S'assurer que la colonne english_text existe
    cursor.execute("PRAGMA table_info(translations_log);")
    columns = [col[1] for col in cursor.fetchall()]
    if 'english_text' not in columns:
        cursor.execute("ALTER TABLE translations_log ADD COLUMN english_text TEXT;")
        conn.commit()
        
    cursor.execute("SELECT id, translated_text FROM translations_log")
    rows = cursor.fetchall()
    print(f" -> Génération des traductions en anglais pour {len(rows)} lignes...")
    
    for row_id, translit in rows:
        words = translit.split()
        translated_words = [LEXICON_MAP.get(w, w) for w in words]
        eng_text = " | ".join(translated_words)
        cursor.execute("UPDATE translations_log SET english_text = ? WHERE id = ?", (eng_text, row_id))
        
    conn.commit()
    
    # Afficher un échantillon pour vérifier
    cursor.execute("SELECT image_name, translated_text, english_text FROM translations_log ORDER BY RANDOM() LIMIT 3")
    samples = cursor.fetchall()
    print("\n[Vérification] Échantillon enrichi (Translittération & Anglais) :\n" + "="*75)
    for img, trans, eng in samples:
        print(f" 🗂️ Identifiant : {img}")
        print(f" 📜 Translit.   : {trans}")
        print(f" 🇬🇧 Anglais     : {eng}")
        print("-" * 75)
        
    conn.close()

if __name__ == "__main__":
    enrich_english()
