import os
import cv2
import sqlite3
import pandas as pd

DB_PATH = "/workspace/CUNEISCANIA/cuneiscania.db"
DATASET_DIR = "/workspace/CUNEISCANIA/ocr_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def extract_crops_for_training():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filepath FROM tablet_photos LIMIT 50") # Traiter 50 tablettes pour le dataset initial
    rows = cursor.fetchall()
    conn.close()

    crop_metadata = []
    crop_count = 0

    for row in rows:
        img_path = row[0]
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 25 < w < 100 and 25 < h < 100:
                crop = img[y:y+h, x:x+w]
                crop_fname = f"crop_{crop_count:05d}.png"
                crop_fpath = os.path.join(DATASET_DIR, crop_fname)
                cv2.imwrite(crop_fpath, crop)

                crop_metadata.append({
                    "filename": crop_fname,
                    "filepath": crop_fpath,
                    "width": w,
                    "height": h,
                    "label": "unlabeled"  # À annoter ou étiqueter automatiquement
                })
                crop_count += 1

    df = pd.DataFrame(crop_metadata)
    df.to_csv("/workspace/CUNEISCANIA/ocr_dataset.csv", index=False)
    print(f"[Dataset] {crop_count} images de signes extraites et sauvegardées dans {DATASET_DIR}")

if __name__ == "__main__":
    extract_crops_for_training()
