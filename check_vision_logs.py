import sqlite3

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"

def check_vision_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, input_text, translated_text, timestamp FROM translations_log ORDER BY RANDOM() LIMIT 5")
    rows = cursor.fetchall()
    
    print("\n[Vérification Vision ResNet18] Échantillon de 5 lignes analysées :\n" + "="*65)
    for img_name, in_path, trans_text, ts in rows:
        print(f" 🗂️  Identifiant  : {img_name}")
        print(f" 📂 Chemin Crop  : {in_path}")
        print(f" 👁️  Translit. Vision : {trans_text}")
        print(f" ⏰ Timestamp    : {ts}")
        print("-" * 65)
    conn.close()

if __name__ == "__main__":
    check_vision_logs()
