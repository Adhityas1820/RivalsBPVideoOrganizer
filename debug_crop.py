"""
debug_crop.py
-------------
Saves the first frame of each video in unsorted_videos/ with three boxes drawn:
  - SLOT box    (white)  : the overall slot region
  - LABEL box   (green)  : sub-region for LSHIFT label detection
  - FILL box    (cyan)   : sub-region that fills white after a dash (test_dash_counter3)

Adjust coordinates below, run, then open debug_crops/ to review.
"""

import cv2
from pathlib import Path

# ── Slot box (overall region) ────────────────────────────────────────────────
SLOT_X0, SLOT_X1, SLOT_Y0, SLOT_Y1 = 25, 600, 900, 1060

# ── Label sub-box (LSHIFT text area) ─────────────────────────────────────────
LABEL_X0, LABEL_X1, LABEL_Y0, LABEL_Y1 = 1450, 1875, 900, 1065

# ── Fill sub-box (fills white after a dash — test_dash_counter3) ─────────────
FILL_X0, FILL_X1,FILL_Y0, FILL_Y1 = 750, 1175, 970, 1050

# ─────────────────────────────────────────────────────────────────────────────

VIDEO_IN = "dataset_videos\Convergence\Symbiotic Surface\Marvel Rivals 2026.03.08 - 20.48.12.07.mp4"
SEEK_SEC = 8
OUT_DIR  = "debug_crops"

Path(OUT_DIR).mkdir(exist_ok=True)

if True:
    video_path = Path(VIDEO_IN)
    cap = cv2.VideoCapture(VIDEO_IN)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(SEEK_SEC * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame at {SEEK_SEC}s from {VIDEO_IN}")
        exit()

    h, w = frame.shape[:2]
    print(f"\n{video_path.name}  ({w}x{h})  @ {SEEK_SEC}s")

    annotated = frame.copy()

    # Slot box — white
    cv2.rectangle(annotated, (SLOT_X0, SLOT_Y0), (SLOT_X1, SLOT_Y1), (255, 255, 255), 2)
    cv2.putText(annotated, "SLOT", (SLOT_X0, SLOT_Y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Label box — green
    cv2.rectangle(annotated, (LABEL_X0, LABEL_Y0), (LABEL_X1, LABEL_Y1), (0, 255, 0), 2)
    cv2.putText(annotated, "LABEL", (LABEL_X0, LABEL_Y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Fill box — cyan
    cv2.rectangle(annotated, (FILL_X0, FILL_Y0), (FILL_X1, FILL_Y1), (0, 220, 220), 2)
    cv2.putText(annotated, "FILL", (FILL_X0, FILL_Y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    out_path = Path(OUT_DIR) / f"{video_path.stem}_boxes.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved : {out_path.name}")

print(f"\nDone. Open debug_crops/ to review.")
