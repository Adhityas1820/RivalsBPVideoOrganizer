"""
assist_kill_counter.py
----------------------
One-shot helper: samples a frame for each of the 6 kill feed rows,
finds the white contours in each row region, saves them all to a
single .npy file, and outputs annotated images for verification.

Set KILL_VIDEO and one SAMPLE_SEC per row to a timestamp where that
row has a kill visible in it.

Outputs:
    models/kill_row_contours.npy       (list of 6 contour arrays)
    test/kill_row_<n>_verify.jpg       (one image per row)

Usage:
    python assist_kill_counter.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
KILL_VIDEO = r"C:\Adhitya\Coding\MarvelRivalsClassifier\unsorted_videos\Hells Heaven - 0d - 1k.mkv"

# Set each sample_sec to a timestamp (in seconds) where that row shows a kill.
# Rows are numbered top to bottom (0 = topmost kill feed entry).
ROWS = [
    {"label": "row0", "search": (1870, 1880,  32,  48), "sample_sec": 5.0},
    {"label": "row1", "search": (1871, 1882,  70,  83), "sample_sec": 5.0},
    {"label": "row2", "search": (1870, 1882, 107, 120), "sample_sec": 5.0},
    {"label": "row3", "search": (1870, 1883, 148, 161), "sample_sec": 5.0},
    {"label": "row4", "search": (1870, 1882, 187, 200), "sample_sec": 5.0},
    {"label": "row5", "search": (1870, 1882, 226, 239), "sample_sec": 5.0},
]

SAVE_PATH    = "models/kill_row_contours.npy"
WHITE_THRESH = 200
ZOOM         = 8
# ---------------------------------------------------------------------------


def find_contours_in_region(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    region = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    return [c + np.array([[[x0, y0]]]) for c in contours]


def main():
    Path("test").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(KILL_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {KILL_VIDEO}")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    all_row_contours = []

    for row in ROWS:
        label      = row["label"]
        x0, x1, y0, y1 = row["search"]
        sample_sec = row["sample_sec"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_sec * src_fps))
        ret, frame = cap.read()
        if not ret:
            print(f"[{label}] ERROR: Could not read frame at {sample_sec}s")
            all_row_contours.append([])
            continue

        fh, fw = frame.shape[:2]
        contours = find_contours_in_region(frame, x0, x1, y0, y1)
        all_row_contours.append(contours)

        if contours:
            print(f"[{label}] Found {len(contours)} contour(s) at t={sample_sec}s")
        else:
            print(f"[{label}] WARNING: No white contours found at t={sample_sec}s — check sample_sec")

        # Annotated image
        annotated = frame.copy()
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (60, 60, 60), 1)
        cv2.putText(annotated, f"{label}  t={sample_sec}s",
                    (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if contours:
            cv2.drawContours(annotated, [c.astype(np.int32) for c in contours], -1, (0, 255, 0), 1)
            rx, ry, rw, rh = cv2.boundingRect(np.concatenate(contours).astype(np.int32))
            cv2.rectangle(annotated, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)

        # Zoomed crop — top right
        crop = frame[min(y0,fh):min(y1,fh), min(x0,fw):min(x1,fw)].copy()
        if crop.size > 0:
            zh = max(crop.shape[0] * ZOOM, 1)
            zw = max(crop.shape[1] * ZOOM, 1)
            zoomed = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_NEAREST)
            if contours:
                for c in contours:
                    sc = ((c - np.array([[[x0, y0]]])) * np.array([[[zw / max(crop.shape[1],1),
                          zh / max(crop.shape[0],1)]]])).astype(np.int32)
                    cv2.drawContours(zoomed, [sc], -1, (0, 255, 0), 1)
            px, py = fw - zw - 20, 60
            if py + zh <= fh and px >= 0:
                annotated[py:py+zh, px:px+zw] = zoomed
                cv2.rectangle(annotated, (px, py), (px+zw, py+zh), (200, 200, 200), 1)

        img_out = f"test/kill_row_{label}_verify.jpg"
        cv2.imwrite(img_out, annotated)
        print(f"[{label}] Saved verification image to {img_out}")

    cap.release()

    # Save all 6 row contour lists
    np.save(SAVE_PATH, np.array(all_row_contours, dtype=object))
    found = sum(1 for c in all_row_contours if c)
    print(f"\nSaved {found}/6 rows with contours to {SAVE_PATH}")
    if found < 6:
        print("  Rows with no contours will use full bounding-box white ratio as fallback.")


if __name__ == "__main__":
    main()
