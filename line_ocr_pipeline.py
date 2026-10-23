import os
import cv2
import sqlite3
import numpy as np

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
LINES_DIR = "/workspace/CUNEISCANIA/tablet_lines"
os.makedirs(LINES_DIR, exist_ok=True)

def extract_text_lines():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filepath FROM tablet_photos LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    print(f"[Line OCR] Analyse de {len(rows)} tablettes...")
    total_lines = 0

    for idx, row in enumerate(rows):
        img_path = row[0]
        if not os.path.exists(img_path):
            print(f" -> Fichier introuvable : {img_path}")
            continue

        tablet_id = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f" -> Impossible de lire l'image : {img_path}")
            continue

        # Redimensionner pour homogénéiser si l'image est immense
        h_orig, w_orig = img.shape
        print(f" -> Tablette {tablet_id} : dimensions {w_orig}x{h_orig}")

        # Utilisation d'un seuillage adaptatif local (plus robuste sur les photos d'antiquités)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)

        # Projection horizontale (somme des pixels d'encre par ligne)
        row_sums = np.sum(thresh, axis=1) / 255.0
        
        # Seuil dynamique adaptatif basé sur la largeur de l'image
        threshold_val = w_orig * 0.01
        
        in_line = False
        line_start = 0
        tablet_lines_dir = os.path.join(LINES_DIR, tablet_id)
        os.makedirs(tablet_lines_dir, exist_ok=True)

        line_idx = 0
        for y, val in enumerate(row_sums):
            if val > threshold_val and not in_line:
                in_line = True
                line_start = max(0, y - 5)
            elif val <= threshold_val and in_line:
                in_line = False
                line_end = min(h_orig, y + 5)
                
                if (line_end - line_start) > 10:  # Hauteur minimale de ligne
                    line_img = img[line_start:line_end, :]
                    line_filename = f"line_{line_idx:02d}.png"
                    cv2.imwrite(os.path.join(tablet_lines_dir, line_filename), line_img)
                    line_idx += 1
                    total_lines += 1

        print(f"    -> Lignes trouvées pour {tablet_id} : {line_idx}")

    print(f"\n[Succès] Total de {total_lines} lignes de texte extraites dans : {LINES_DIR}/")

if __name__ == "__main__":
    extract_text_lines()
