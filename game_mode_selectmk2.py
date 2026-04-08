"""
train_game_modemk2.py
---------------------
Rule-based game mode classifier. Processes videos in unsorted_videos/ and
classifies each as Domination, Convoy, or Convergence.

Logic:
  1. Check two slots for rgb(32,28,38) → Domination
  2. Check a third slot for blue, orange, or white → Convoy
  3. Otherwise → Convergence

Usage:
    python train_game_modemk2.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — Domination check (both slots must match the dark color)
# ---------------------------------------------------------------------------
DOM_SLOT1 = (1075, 1100, 50, 65)
DOM_SLOT2 = ( 820,  845, 50, 65)

DOM_TARGET_BGR = np.array([38, 28, 32], dtype=np.int16)  # rgb(32,28,38) → BGR
DOM_TOL        = 40    # per-channel tolerance
DOM_RATIO      = 0.9   # fraction of pixels that must match

# ---------------------------------------------------------------------------
# Config — Convoy check (slot must contain blue, orange, or white)
# ---------------------------------------------------------------------------
CONVOY_SLOT = (950, 960, 105, 110)

# Blue: blue channel dominant
BLUE_MIN_B   = 150
BLUE_B_R_GAP = 60   # B - R must exceed this

# Orange: red channel dominant
ORANGE_MIN_R   = 150
ORANGE_R_B_GAP = 80  # R - B must exceed this

# Black: same target/tolerance as domination slots
# (reuses DOM_TARGET_BGR and DOM_TOL)

CONVOY_RATIO = 0.3   # fraction of pixels that must be blue/orange/black

# ---------------------------------------------------------------------------
# Config — sampling + voting
# ---------------------------------------------------------------------------
SAMPLE_FRAMES    = 10
DOMINATION_VOTE  = 0.6
CONVOY_VOTE      = 0.5

DEBUG_DIR        = "test"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# ---------------------------------------------------------------------------


def crop(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    return frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]


def slot_matches_dom(frame, region) -> bool:
    c = crop(frame, *region)
    if c.size == 0:
        return False
    diff = np.abs(c.astype(np.int16) - DOM_TARGET_BGR).max(axis=2)
    return (diff <= DOM_TOL).sum() / max(c.shape[0] * c.shape[1], 1) >= DOM_RATIO


def slot_is_convoy_color(frame, region) -> tuple:
    """Returns (is_convoy, breakdown_dict)."""
    c = crop(frame, *region)
    if c.size == 0:
        return False, {}

    b = c[:,:,0].astype(np.int16)
    g = c[:,:,1].astype(np.int16)
    r = c[:,:,2].astype(np.int16)
    total = max(c.shape[0] * c.shape[1], 1)

    blue   = ((b > BLUE_MIN_B) & ((b - r) > BLUE_B_R_GAP))
    orange = ((r > ORANGE_MIN_R) & ((r - b) > ORANGE_R_B_GAP))
    diff   = np.abs(c.astype(np.int16) - DOM_TARGET_BGR).max(axis=2)
    black  = diff <= DOM_TOL
    convoy = blue | orange | black

    breakdown = {
        "blue":   blue.sum()   / total,
        "orange": orange.sum() / total,
        "black":  black.sum()  / total,
        "convoy": convoy.sum() / total,
    }
    return breakdown["convoy"] >= CONVOY_RATIO, breakdown


def slot_avg_rgb(frame, region):
    c = crop(frame, *region)
    if c.size == 0:
        return 0, 0, 0
    avg = c.reshape(-1, 3).mean(axis=0)
    return int(avg[2]), int(avg[1]), int(avg[0])  # r, g, b


def save_debug_frame(video_path, frame, t_sec, idx, is_dom, is_convoy, dom_b1, dom_b2, breakdown):
    out_dir = Path(DEBUG_DIR) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = frame.copy()

    for region, label, matched in [(DOM_SLOT1, "dom_slot1", dom_b1), (DOM_SLOT2, "dom_slot2", dom_b2)]:
        x0, x1, y0, y1 = region
        r, g, b = slot_avg_rgb(frame, region)
        color = (0, 200, 0) if matched else (0, 0, 200)
        cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
        cv2.putText(ann, f"{label} rgb({r},{g},{b}) {'[MATCH]' if matched else '[NO]'}",
                    (x0 - 10, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    x0, x1, y0, y1 = CONVOY_SLOT
    r, g, b = slot_avg_rgb(frame, CONVOY_SLOT)
    color = (0, 200, 0) if is_convoy else (0, 0, 200)
    cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
    bd = breakdown
    cv2.putText(ann,
                f"convoy rgb({r},{g},{b}) Bk:{bd.get('black',0)*100:.0f}% B:{bd.get('blue',0)*100:.0f}% O:{bd.get('orange',0)*100:.0f}%",
                (x0 - 250, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if is_dom:
        mode = "DOMINATION"
    elif is_convoy:
        mode = "CONVOY"
    else:
        mode = "CONVERGENCE"
    cv2.putText(ann, f"{mode}  t={t_sec:.1f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out = str(out_dir / f"{idx:02d}.jpg")
    cv2.imwrite(out, ann)
    print(f"    Saved: {out}")


def classify_video(video_path: Path, debug: bool = False) -> tuple:
    """
    Returns (mode_str, dom_confidence, convoy_confidence).
    mode_str: 'Domination', 'Convoy', or 'Convergence'.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "Unknown", 0.0, 0.0

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step         = max(1, total_frames // SAMPLE_FRAMES)

    dom_hits    = 0
    convoy_hits = 0
    sampled     = 0
    read_idx    = 0

    while sampled < SAMPLE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % step == 0:
            dom_b1   = slot_matches_dom(frame, DOM_SLOT1)
            dom_b2   = slot_matches_dom(frame, DOM_SLOT2)
            is_dom   = dom_b1 and dom_b2

            is_convoy, breakdown = slot_is_convoy_color(frame, CONVOY_SLOT)

            if is_dom:
                dom_hits += 1
            if is_convoy:
                convoy_hits += 1

            t_sec = read_idx / src_fps
            r, g, b = slot_avg_rgb(frame, CONVOY_SLOT)
            bd = breakdown
            print(f"  frame {sampled+1:02d} @ {t_sec:.1f}s"
                  f"  dom:[{'V' if dom_b1 else 'X'} {'V' if dom_b2 else 'X'}]"
                  f"  convoy_slot rgb({r},{g},{b})"
                  f"  Bk:{bd.get('black',0)*100:.0f}%"
                  f"  B:{bd.get('blue',0)*100:.0f}%"
                  f"  O:{bd.get('orange',0)*100:.0f}%"
                  f"  -> {'DOM' if is_dom else 'CONVOY' if is_convoy else 'conv'}")

            if debug:
                save_debug_frame(video_path, frame, t_sec, sampled,
                                 is_dom, is_convoy, dom_b1, dom_b2, breakdown)
            sampled += 1
        read_idx += 1

    cap.release()

    dom_conf    = dom_hits    / max(sampled, 1)
    convoy_conf = convoy_hits / max(sampled, 1)

    if dom_conf >= DOMINATION_VOTE:
        mode = "Domination"
    elif convoy_conf >= CONVOY_VOTE:
        mode = "Convoy"
    else:
        mode = "Convergence"

    return mode, dom_conf, convoy_conf


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
        mode, dom_conf, convoy_conf = classify_video(v, debug=True)
        print(f"\n  >> Result: {mode.upper()}"
              f"  (dom={dom_conf*100:.0f}%  convoy={convoy_conf*100:.0f}%)\n")


if __name__ == "__main__":
    main()
