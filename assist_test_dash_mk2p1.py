"""
assist_test_dash_mk2p1.py
--------------------------
One-shot helper: samples the dash icon contour from each slot's source video,
saves them to .npy files, and outputs annotated images for verification.

Right slot (slot2): unsorted_videos/arakko 3.mp4  @ 7.95s
Left  slot (slot3): unsorted_videos/krakoa.mp4     @ 12.04s

Outputs:
    test/contour_verify_right.jpg
    test/contour_verify_left.jpg

Usage:
    python assist_test_dash_mk2p1.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
RIGHT_VIDEO      = "unsorted_videos/arakko 3.mp4"
RIGHT_SAMPLE_SEC = 7.95
RIGHT_SEARCH     = (1575, 1625, 965, 1000)   # (x0, x1, y0, y1)
RIGHT_SAVE_PATH  = "reference pictures/slot_x_contour_right.npy"
RIGHT_IMAGE_OUT  = "test/contour_verify_right.jpg"

LEFT_VIDEO       = "unsorted_videos/krakoa.mp4"
LEFT_SAMPLE_SEC  = 17.33
LEFT_SEARCH      = (1500, 1550, 965, 1000)
LEFT_SAVE_PATH   = "reference pictures/slot_x_contour_left.npy"
LEFT_IMAGE_OUT   = "test/contour_verify_left.jpg"

WHITE_THRESH = 200
ZOOM         = 10
# ---------------------------------------------------------------------------


def find_all_contours(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    region = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    return [c + np.array([[[x0, y0]]]) for c in contours]


def process(video_path, sample_sec, search, save_path, image_out, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_sec * src_fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Could not read frame at {sample_sec}s from {video_path}")
        return

    x0, x1, y0, y1 = search
    contours = find_all_contours(frame, x0, x1, y0, y1)

    if contours:
        np.save(save_path, np.array(contours, dtype=object))
        print(f"[{label}] Saved {len(contours)} contour(s) to {save_path}")
    else:
        print(f"[{label}] WARNING: No white contours found — .npy not saved")

    # --- Annotate frame ---
    fh, fw = frame.shape[:2]

    # Search region
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

    # Contours
    if contours:
        rx, ry, rw, rh = cv2.boundingRect(np.concatenate(contours).astype(np.int32))
        cv2.drawContours(frame, [c.astype(np.int32) for c in contours], -1, (0, 255, 0), 1)
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
        cv2.putText(frame, f"{label} ({len(contours)} contours)", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"{label} NO CONTOURS", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # Zoomed preview — top right
    crop = frame[min(y0,fh):min(y1,fh), min(x0,fw):min(x1,fw)].copy()
    if crop.size > 0:
        zh = max(crop.shape[0] * ZOOM, 1)
        zw = max(crop.shape[1] * ZOOM, 1)
        zoomed = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_NEAREST)
        if contours:
            for c in contours:
                sc = ((c - np.array([[[x0, y0]]])) * np.array([[[zw / max(crop.shape[1],1), zh / max(crop.shape[0],1)]]])).astype(np.int32)
                cv2.drawContours(zoomed, [sc], -1, (0, 255, 0), 1)
        px, py = fw - zw - 20, 60
        if py + zh <= fh and px >= 0:
            frame[py:py+zh, px:px+zw] = zoomed
            cv2.rectangle(frame, (px, py), (px+zw, py+zh), (200, 200, 200), 1)
            cv2.putText(frame, f"{label} zoomed", (px, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, f"{label}  t={sample_sec}s  src={video_path}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(image_out, frame)
    print(f"[{label}] Saved verification image to {image_out}")


def main():
    Path("test").mkdir(exist_ok=True)
    Path("reference pictures").mkdir(exist_ok=True)

    process(RIGHT_VIDEO, RIGHT_SAMPLE_SEC, RIGHT_SEARCH,
            RIGHT_SAVE_PATH, RIGHT_IMAGE_OUT, "RIGHT/slot2")

    process(LEFT_VIDEO,  LEFT_SAMPLE_SEC,  LEFT_SEARCH,
            LEFT_SAVE_PATH,  LEFT_IMAGE_OUT,  "LEFT/slot3")


if __name__ == "__main__":
    main()
