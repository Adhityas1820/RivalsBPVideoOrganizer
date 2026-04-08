"""
game_mode_select.py
-------------------
Detects whether a video is Domination mode by checking if two HUD slots
are both black. If both are black → Domination, otherwise → Not Domination.

Usage:
    python game_mode_select.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1 = 1075, 1100, 50, 65
SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1 =  820,  845, 50, 65

TARGET_BGR      = np.array([38, 28, 32], dtype=np.int16)  # rgb(32,28,38) → BGR
COLOR_TOL       = 40    # per-channel tolerance
COLOR_RATIO     = 0.9  # fraction of pixels matching the color → slot is "black"
SAMPLE_FRAMES   = 10    # number of evenly spaced frames to sample
DOMINATION_VOTE = 0.6   # fraction of sampled frames where both slots are black → Domination

DEBUG_DIR       = "test"
ZOOM            = 8     # zoom factor for debug crops

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# ---------------------------------------------------------------------------


def slot_is_black(frame, x0, x1, y0, y1) -> bool:
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return False
    diff = np.abs(crop.astype(np.int16) - TARGET_BGR).max(axis=2)
    return (diff <= COLOR_TOL).sum() / max(crop.shape[0] * crop.shape[1], 1) >= COLOR_RATIO


def slot_stats(frame, x0, x1, y0, y1):
    """Return (avg_bgr, avg_dist_from_target, match_ratio) for a slot region."""
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return (0, 0, 0), 0.0, 0.0
    avg_bgr    = crop.reshape(-1, 3).mean(axis=0)
    diff       = np.abs(crop.astype(np.int16) - TARGET_BGR).max(axis=2)
    avg_dist   = diff.mean()
    match_ratio = (diff <= COLOR_TOL).sum() / max(crop.shape[0] * crop.shape[1], 1)
    return avg_bgr, avg_dist, match_ratio


def save_debug_frame(video_path: Path, frame, t_sec: float, idx: int, b1: bool, b2: bool):
    """Save one annotated full frame to test/."""
    Path(DEBUG_DIR).mkdir(exist_ok=True)
    annotated = frame.copy()
    h, w = frame.shape[:2]

    for x0, x1, y0, y1, label, black in [
        (SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1, "slot1", b1),
        (SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1, "slot2", b2),
    ]:
        avg_bgr, avg_dist, match_ratio = slot_stats(frame, x0, x1, y0, y1)
        r, g, b = int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])
        color = (0, 0, 200) if black else (0, 200, 0)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        cv2.putText(annotated, f"{label}: rgb({r},{g},{b})  dist:{avg_dist:.0f}  match:{match_ratio*100:.0f}%",
                    (x0 - 200, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        print(f"    {label}: avg=rgb({r},{g},{b})  dist={avg_dist:.1f}  match={match_ratio*100:.0f}%  {'[MATCH]' if black else '[NO MATCH]'}")

    mode = "DOMINATION" if (b1 and b2) else "NOT DOMINATION"
    cv2.putText(annotated, f"{mode}  t={t_sec:.1f}s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out = str(Path(DEBUG_DIR) / f"{video_path.stem}_gamemode_{idx:02d}.jpg")
    cv2.imwrite(out, annotated)
    print(f"    Saved: {out}")


def is_domination(video_path: Path, debug: bool = False) -> tuple:
    """
    Returns (is_domination: bool, confidence: float).
    Samples SAMPLE_FRAMES evenly across the video sequentially (fast).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0.0

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step         = max(1, total_frames // SAMPLE_FRAMES)

    both_black = 0
    sampled    = 0
    read_idx   = 0

    while sampled < SAMPLE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % step == 0:
            b1 = slot_is_black(frame, SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1)
            b2 = slot_is_black(frame, SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1)
            if b1 and b2:
                both_black += 1
            if debug:
                save_debug_frame(video_path, frame, read_idx / src_fps, sampled, b1, b2)
            sampled += 1
        read_idx += 1

    cap.release()
    confidence = both_black / max(sampled, 1)
    return confidence >= DOMINATION_VOTE, confidence


def main():
    input_dir = Path("unsorted_videos")
    videos = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not videos:
        print("No videos found in unsorted_videos/")
        return

    for v in videos:
        print(f"\n{'='*60}")
        print(f"  {v.name}")
        print(f"{'='*60}")
        dom, conf = is_domination(v, debug=True)
        mode = "DOMINATION" if dom else "NOT DOMINATION"
        print(f"\n  >> Result: {mode}  (confidence: {conf*100:.0f}%)\n")


if __name__ == "__main__":
    main()
