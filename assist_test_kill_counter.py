"""
assist_test_kill_counter.py
----------------------------
One-shot helper: samples a frame at the moment a kill appears,
finds the white contours in the kill slot region, saves them to a
.npy file, and outputs an annotated image for verification.

Same approach as assist_test_dash_mk2p1.py.

Set KILL_VIDEO and KILL_SAMPLE_SEC to a frame where a kill is visible,
and KILL_SEARCH to the region where the kill indicator lights up.

Outputs:
    models/kill_slot_contour.npy
    test/kill_slot_verify.jpg

Usage:
    python assist_test_kill_counter.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
KILL_VIDEO      = r"unsorted_videos\celestial husk 1 - 5 kills - 9 dashes.mp4"
KILL_SAMPLE_SEC = 0.0           # <- set to a timestamp where the scoreboard is visible

SLOTS = [
    {
        "search":   (760,  938,  232, 278),
        "save":     "models/scoreboard_slot1_contour.npy",
        "img_out":  "test/scoreboard_slot1_verify.jpg",
        "label":    "slot1",
    },
    {
        "search":   (1423, 1597, 236, 283),
        "save":     "models/scoreboard_slot2_contour.npy",
        "img_out":  "test/scoreboard_slot2_verify.jpg",
        "label":    "slot2",
    },
]

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


def main():
    Path("test").mkdir(exist_ok=True)
    Path("reference pictures").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(KILL_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {KILL_VIDEO}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(KILL_SAMPLE_SEC * src_fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Could not read frame at {KILL_SAMPLE_SEC}s")
        return

    fh, fw = frame.shape[:2]
    cv2.putText(frame, f"t={KILL_SAMPLE_SEC}s  src={KILL_VIDEO}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    for slot in SLOTS:
        x0, x1, y0, y1 = slot["search"]
        contours = find_all_contours(frame, x0, x1, y0, y1)

        if contours:
            np.save(slot["save"], np.array(contours, dtype=object))
            print(f"[{slot['label']}] Saved {len(contours)} contour(s) to {slot['save']}")
        else:
            print(f"[{slot['label']}] WARNING: No white contours found — .npy not saved")

        # --- Annotate frame ---
        annotated = frame.copy()
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (60, 60, 60), 1)

        if contours:
            rx, ry, rw, rh = cv2.boundingRect(np.concatenate(contours).astype(np.int32))
            cv2.drawContours(annotated, [c.astype(np.int32) for c in contours], -1, (0, 255, 0), 1)
            cv2.rectangle(annotated, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
            cv2.putText(annotated, f"{slot['label']} ({len(contours)} contours)", (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(annotated, f"{slot['label']} NO CONTOURS", (x0, y0 - 5),
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
                annotated[py:py+zh, px:px+zw] = zoomed
                cv2.rectangle(annotated, (px, py), (px+zw, py+zh), (200, 200, 200), 1)
                cv2.putText(annotated, f"{slot['label']} zoomed", (px, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imwrite(slot["img_out"], annotated)
        print(f"[{slot['label']}] Saved verification image to {slot['img_out']}")


if __name__ == "__main__":
    main()
