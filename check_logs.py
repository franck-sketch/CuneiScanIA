import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
REPORT_OUTPUT = "/workspace/CUNEISCANIA/pipeline_summary_stage3.json"

def check_and_centralize_logs():
    print(f"\n[Monitoring] Connexion à la base de données : {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print(f"[Attention] La base de données {DB_PATH} est introuvable. Vérifie son emplacement.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Vérification des tables disponibles dans la base
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"[Monitoring] Tables détectées dans la base : {tables}")

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "database": DB_PATH,
        "metrics": {}
    }

    # Exemple d'analyse dynamique si une table de logs ou de résultats existe
    if "execution_logs" in tables or "predictions" in tables:
        target_table = "execution_logs" if "execution_logs" in tables else "predictions"
        
        cursor.execute(f"SELECT COUNT(*) FROM {target_table};")
        total_records = cursor.fetchone()[0]
        report_data["metrics"]["total_records"] = total_records
        
        print(f" - Nombre total d'enregistrements dans '{target_table}' : {total_records}")
    else:
        print(" - Aucune table de logs standard ('execution_logs' ou 'predictions') trouvée. Analyse de structure vide.")
        report_data["metrics"]["total_records"] = 0

    conn.close()

    # Sauvegarde du rapport unifié
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f"\n[Succès] Rapport de monitoring généré avec succès : {REPORT_OUTPUT}")

if __name__ == "__main__":
    check_and_centralize_logs()
