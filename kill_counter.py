"""
kill_counter.py
---------------
Counts kills in Marvel Rivals gameplay clips by monitoring stacked rows
of the kill feed region. A row is "lit" when its white pixel ratio exceeds
ROW_WHITE_RATIO AND the two largest white contours have a closest-point
distance between DIST_MIN and DIST_MAX pixels.

Scoreboard awareness: when the scoreboard is open (detected via contour-based
slot matching), variables update normally but no kills are counted. A 0.2s
lockout after the board closes prevents false kills from rows that were already
lit before it opened.

Outputs total kills + timestamps for each video, and saves a summary JSON.

Usage:
    python kill_counter.py
"""

import json
import shutil
import tempfile
import multiprocessing as mp
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR  = "unsorted_videos"
OUTPUT_DIR = "kill_counts"

PROCESS_FPS = 60

X_START    = 1625
X_END      = 1875
ROW_H      = 30
ROW_GAP    = 10
ROW_COUNT  = 6
ROW_Y_BASE = 30

ROW_Y_STARTS = [ROW_Y_BASE + i * (ROW_H + ROW_GAP) for i in range(ROW_COUNT)]

WHITE_THRESH    = 200
ROW_WHITE_RATIO = 0.20

DIST_MIN = 2
DIST_MAX = 10

STABLE_FRAMES    = 10
KILL_TIMER_SECS  = 5.15

# Scoreboard slots
SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1 = 760,  938,  232, 278
SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1 = 1423, 1597, 236, 283

SLOT1_CONTOUR_PATH = "models/scoreboard_slot1_contour.npy"
SLOT2_CONTOUR_PATH = "models/scoreboard_slot2_contour.npy"

SLOT_WHITE_RATIO = 0.80
BOX_MAX_RATIO    = 0.30
BOARD_LOCKOUT    = 0.2

# Ability indicator — triggers a wider DIST_MAX window on yellow→blue transition
ABILITY_X0, ABILITY_X1, ABILITY_Y0, ABILITY_Y1 = 1865, 1870, 943, 949

YELLOW_R_MIN, YELLOW_G_MIN, YELLOW_B_MAX = 150, 150, 100
BLUE_B_MIN = 200
ABILITY_COLOR_THRESH = 0.90

DIST_MAX_WIDE      = 90
WIDE_DELAY_SECS    = 1
WIDE_DURATION_SECS = 0.7

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# ---------------------------------------------------------------------------


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    tmp = tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    return cv2.VideoCapture(tmp.name), tmp.name


def load_contours(path):
    p = Path(path)
    if not p.exists():
        return []
    return list(np.load(str(p), allow_pickle=True))


def row_is_lit(crop: np.ndarray, dist_max: int = DIST_MAX) -> bool:
    b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    white_mask = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
    ratio = white_mask.sum() / white_mask.size
    if ratio < ROW_WHITE_RATIO:
        return False
    mask = white_mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top2 = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    if len(top2) < 2:
        return False
    pts1 = top2[0].reshape(-1, 2).astype(np.float32)
    pts2 = top2[1].reshape(-1, 2).astype(np.float32)
    dist = float(np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2).min())
    return DIST_MIN <= dist <= dist_max


def slot_is_white(frame, contours, x0, x1, y0, y1) -> bool:
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return False
    b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    white = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
    box_ratio = white.sum() / white.size
    if contours:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        shifted = [(c - np.array([[[x0, y0]]])).astype(np.int32) for c in contours]
        cv2.drawContours(mask, shifted, -1, 255, thickness=cv2.FILLED)
        total = mask.sum() // 255
        if total == 0:
            return False
        contour_ratio = ((white) & (mask > 0)).sum() / total
    else:
        contour_ratio = box_ratio
    return contour_ratio >= SLOT_WHITE_RATIO and box_ratio <= BOX_MAX_RATIO


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def count_kills(video_path: Path, contours1: list, contours2: list) -> tuple:
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return video_path.name, 0, []

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / PROCESS_FPS))

    pending_kill_timers = []
    candidate_count     = 0
    streak              = 0
    total_kills         = 0
    timestamps          = []
    board_on            = False
    kill_locked_until   = 0.0
    read_idx            = 0

    ability_is_yellow  = False
    wide_dist_start_at = -1.0
    wide_dist_end_at   = -1.0
    cur_dist_max       = DIST_MAX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx % frame_interval == 0:
            hf, wf = frame.shape[:2]
            t_sec  = read_idx / src_fps

            # Ability indicator: detect yellow→blue transition
            ab_crop = frame[min(ABILITY_Y0,hf):min(ABILITY_Y1,hf),
                            min(ABILITY_X0,wf):min(ABILITY_X1,wf)]
            if ab_crop.size > 0:
                b_ab, g_ab, r_ab = ab_crop[:,:,0], ab_crop[:,:,1], ab_crop[:,:,2]
                n_px = ab_crop.shape[0] * ab_crop.shape[1]
                yellow_r = ((r_ab > YELLOW_R_MIN) & (g_ab > YELLOW_G_MIN) & (b_ab < YELLOW_B_MAX)).sum() / n_px
                blue_r   = ((b_ab > BLUE_B_MIN) & (b_ab > r_ab) & (b_ab > g_ab)).sum() / n_px
                is_yellow_now = yellow_r >= ABILITY_COLOR_THRESH
                is_blue_now   = blue_r   >= ABILITY_COLOR_THRESH
                if ability_is_yellow and is_blue_now:
                    wide_dist_start_at = t_sec + WIDE_DELAY_SECS
                    wide_dist_end_at   = t_sec + WIDE_DELAY_SECS + WIDE_DURATION_SECS
                ability_is_yellow = is_yellow_now
            cur_dist_max = DIST_MAX_WIDE if (wide_dist_start_at <= t_sec <= wide_dist_end_at) else DIST_MAX

            # Scoreboard detection
            s1 = slot_is_white(frame, contours1, SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1)
            s2 = slot_is_white(frame, contours2, SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1)
            prev_board_on = board_on
            board_on      = s1 or s2
            if prev_board_on and not board_on:
                kill_locked_until = t_sec + BOARD_LOCKOUT

            lit = sum(
                1 for y0 in ROW_Y_STARTS
                if row_is_lit(frame[y0:min(y0+ROW_H,hf), min(X_START,wf):min(X_END,wf)], cur_dist_max)
            )

            if lit == candidate_count:
                streak += 1
            else:
                candidate_count = lit
                streak          = 1

            locked = board_on or t_sec < kill_locked_until

            # Count kills immediately when lit count stabilises with new uncovered rows
            if streak == STABLE_FRAMES and not locked:
                uncovered = candidate_count - len(pending_kill_timers)
                if uncovered > 0:
                    total_kills += uncovered
                    t = fmt_timestamp(t_sec)
                    for _ in range(uncovered):
                        timestamps.append(t)
                        pending_kill_timers.append(t_sec + KILL_TIMER_SECS)

            # Fire expired timers — detect replacement kills (simultaneous fade/replace)
            if not locked:
                fired = sum(1 for e in pending_kill_timers if t_sec >= e)
                if fired > 0:
                    pending_kill_timers[:] = [e for e in pending_kill_timers if t_sec < e]
                    uncovered = candidate_count - len(pending_kill_timers)
                    if uncovered > 0:
                        total_kills += uncovered
                        t = fmt_timestamp(t_sec)
                        for _ in range(uncovered):
                            timestamps.append(t)
                            pending_kill_timers.append(t_sec + KILL_TIMER_SECS)

        read_idx += 1

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return video_path.name, total_kills, timestamps


def _worker(args: tuple) -> tuple:
    video_path_str, contours1, contours2 = args
    return count_kills(Path(video_path_str), contours1, contours2)


def main():
    contours1 = load_contours(SLOT1_CONTOUR_PATH)
    contours2 = load_contours(SLOT2_CONTOUR_PATH)
    print(f"Scoreboard slot 1 : {len(contours1)} contour(s)")
    print(f"Scoreboard slot 2 : {len(contours2)} contour(s)")

    input_path = Path(INPUT_DIR)
    videos = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not videos:
        print(f"No videos found in '{INPUT_DIR}/'.")
        return

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Processing {len(videos)} video(s) at {PROCESS_FPS} fps  |  workers: {num_workers}\n")
    print("=" * 60)

    args = [(str(v), contours1, contours2) for v in videos]
    with mp.Pool(processes=num_workers) as pool:
        all_results = pool.map(_worker, args)

    results = {}
    for name, total, timestamps in all_results:
        times_str = ", ".join(timestamps) if timestamps else "none"
        print(f"[{name}]")
        print(f"  Kills : {total}")
        print(f"  Times : {times_str}\n")
        results[name] = {"kills": total, "timestamps": timestamps}

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUTPUT_DIR) / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
