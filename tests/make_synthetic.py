"""Génère une fausse tablette photographiée, pour tester le pipeline sans données.

Usage : python tests/make_synthetic.py tests/assets/tablet.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def make(out: str | Path, line_angle_deg: float = 8.0, n_lines: int = 8,
         seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h, w = 1200, 1600
    img = np.full((h, w), 35, np.uint8)
    img = cv2.add(img, rng.normal(0, 4, (h, w)).astype(np.int16).astype(np.uint8))

    quad = np.array([[190, 150], [1440, 110], [1470, 1080], [160, 1040]], np.float32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    tablet = np.full((h, w), 176, np.float32)
    yy = np.linspace(0, 1, h)[:, None]
    tablet = tablet - 40 * yy + rng.normal(0, 6, (h, w))
    img = np.where(mask > 0, np.clip(tablet, 0, 255), img).astype(np.uint8)

    ink = np.zeros((h, w), np.uint8)
    theta = np.deg2rad(line_angle_deg)
    x0, y0 = 260, 260
    for li in range(n_lines):
        base_y = y0 + li * 88
        x = x0
        while x < 1320:
            cx, cy = x, base_y + int(np.tan(theta) * (x - x0))
            for k in range(3):  # trois clous par signe
                px = cx + k * 11
                cv2.line(ink, (px, cy), (px + 14, cy + 5), 255, 4, cv2.LINE_AA)
                cv2.line(ink, (px, cy), (px + 3, cy + 13), 255, 3, cv2.LINE_AA)
            x += 46
    img = np.where(ink > 0, np.clip(img.astype(np.int16) - 95, 0, 255), img).astype(np.uint8)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return {"image": str(out), "quad": quad.tolist(), "line_angle_deg": line_angle_deg,
            "n_lines": n_lines}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tests/assets/tablet.png"
    print(make(target))
