"""
dash_counter.py
---------------
Counts dashes in Marvel Rivals gameplay clips using contour-based white
detection + zoom exclusion. Also classifies dash combos (Double/Triple/Quad/Penta)
by tracking when the label turns non-white (ability on cooldown).

Usage:
    python dash_counter.py
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
OUTPUT_DIR = "dash_counts"

PROCESS_FPS = 60

SLOT2_SEARCH = (1575, 1625, 965, 1000)
SLOT3_SEARCH = (1500, 1550, 965, 1000)
SLOT2_LABEL  = (1575, 1625, 1008, 1050)
SLOT3_LABEL  = (1500, 1550, 1008, 1050)
SLOT_DETECT_FRAMES = 30

RIGHT_CONTOUR_PATH = "reference pictures/slot_x_contour_right.npy"
LEFT_CONTOUR_PATH  = "reference pictures/slot_x_contour_left.npy"

WHITE_THRESH       = 200
WHITE_RATIO_THRESH = 1.0
ZOOM_LOW_THRESH    = 0.2
OFF_FRAMES         = 5
DASH_REARM_SECS    = 0.3

LABEL_WHITE_THRESH  = 0.1   # label white ratio below this → ability on cooldown
LABEL_STABLE_FRAMES = 10
COMBO_GAP_SECS      = 0.9
COMBO_NAMES         = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}

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


def label_white_ratio(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (gray > WHITE_THRESH).sum() / max(gray.size, 1)


def white_ratio_in_contours(frame, contours, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    region = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask   = np.zeros(gray.shape, dtype=np.uint8)
    shifted = [(c - np.array([[[x0, y0]]])).astype(np.int32) for c in contours]
    cv2.drawContours(mask, shifted, -1, 255, thickness=cv2.FILLED)
    total_pixels = mask.sum() // 255
    if total_pixels == 0:
        return False, 0.0
    white_pixels = ((gray > WHITE_THRESH) & (mask > 0)).sum()
    ratio = white_pixels / total_pixels
    return ratio >= WHITE_RATIO_THRESH, ratio


def zoom_ratio_excluding_contours(frame, contours, sx0, sy0, sx1, sy1):
    h, w = frame.shape[:2]
    crop = frame[min(sy0,h):min(sy1,h), min(sx0,w):min(sx1,w)]
    if crop.size == 0:
        return 0.0
    gray_z    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    excl_mask = np.zeros(gray_z.shape, dtype=np.uint8)
    if contours:
        shifted = [(c - np.array([[[sx0, sy0]]])).astype(np.int32) for c in contours]
        cv2.drawContours(excl_mask, shifted, -1, 255, thickness=cv2.FILLED)
    outside = excl_mask == 0
    total   = outside.sum()
    return ((gray_z > WHITE_THRESH) & outside).sum() / total if total > 0 else 0.0


def load_contours(path_str):
    p = Path(path_str)
    if not p.exists():
        return []
    data = np.load(str(p), allow_pickle=True)
    return list(data)


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def count_dashes(video_path: Path) -> tuple:
    """
    Returns (video_name, total_dashes, timestamp_strings, combos)
    combos = list of (count, label) e.g. [(3, "Triple"), (2, "Double")]
    """
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return video_path.name, 0, [], []

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / PROCESS_FPS))
    rearm_frames   = int(DASH_REARM_SECS * src_fps)

    # Detect which slot has the LSHIFT label
    ratio2_acc = ratio3_acc = 0.0
    sampled = 0
    while sampled < SLOT_DETECT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        ratio2_acc += label_white_ratio(frame, *SLOT2_LABEL)
        ratio3_acc += label_white_ratio(frame, *SLOT3_LABEL)
        sampled += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    is_right      = ratio2_acc >= ratio3_acc
    slot_contours = load_contours(RIGHT_CONTOUR_PATH if is_right else LEFT_CONTOUR_PATH)
    active_label  = SLOT2_LABEL if is_right else SLOT3_LABEL

    if slot_contours:
        rx, ry, rw, rh = cv2.boundingRect(np.concatenate(slot_contours).astype(np.int32))
        sx0, sx1, sy0, sy1 = rx, rx + rw, ry, ry + rh
    else:
        sx0, sx1, sy0, sy1 = SLOT2_SEARCH if is_right else SLOT3_SEARCH

    # Detection state
    off_streak = 0
    was_off    = True
    rearm_at   = 0
    total_dashes = 0
    timestamps   = []

    # Combo state
    label_is_grey          = False
    label_candidate_grey   = False
    label_candidate_streak = 0
    label_stable_grey      = False
    prev_label_stable_grey = False
    combo_count            = 0
    last_combo_dash_sec    = None
    combos                 = []

    read_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx % frame_interval == 0:
            white_state, _ = white_ratio_in_contours(frame, slot_contours, sx0, sy0, sx1, sy1)
            zoom_ratio     = zoom_ratio_excluding_contours(frame, slot_contours, sx0, sy0, sx1, sy1)
            zoom_low       = zoom_ratio < ZOOM_LOW_THRESH

            if white_state and zoom_low and read_idx >= rearm_at and was_off:
                total_dashes += 1
                t_sec = read_idx / src_fps
                timestamps.append(fmt_timestamp(t_sec))
                rearm_at = read_idx + rearm_frames
                was_off  = False

                if label_stable_grey:
                    if last_combo_dash_sec is None or (t_sec - last_combo_dash_sec) <= COMBO_GAP_SECS:
                        combo_count += 1
                    else:
                        if combo_count >= 2:
                            combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))
                        combo_count = 1
                    last_combo_dash_sec = t_sec

            if not white_state:
                off_streak += 1
                if off_streak >= OFF_FRAMES:
                    was_off = True
            else:
                off_streak = 0

            # Label stable state tracking
            lbl_ratio     = label_white_ratio(frame, *active_label)
            label_is_grey = lbl_ratio < LABEL_WHITE_THRESH
            if label_is_grey == label_candidate_grey:
                label_candidate_streak += 1
            else:
                label_candidate_grey   = label_is_grey
                label_candidate_streak = 1
            prev_label_stable_grey = label_stable_grey
            if label_candidate_streak >= LABEL_STABLE_FRAMES:
                label_stable_grey = label_candidate_grey

            # Label went white → finalize open combo
            if prev_label_stable_grey and not label_stable_grey:
                if combo_count >= 2:
                    combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))
                combo_count         = 0
                last_combo_dash_sec = None

        read_idx += 1

    # Finalize any combo still open at end of video
    if combo_count >= 2:
        combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return video_path.name, total_dashes, timestamps, combos


def _worker(video_path_str: str) -> tuple:
    return count_dashes(Path(video_path_str))


def main():
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

    with mp.Pool(processes=num_workers) as pool:
        all_results = pool.map(_worker, [str(v) for v in videos])

    results = {}
    for name, total, timestamps, combos in all_results:
        times_str  = ", ".join(timestamps) if timestamps else "none"
        combos_str = ", ".join(f"{label} ({n})" for n, label in combos) if combos else "none"
        print(f"[{name}]")
        print(f"  Dashes : {total}")
        print(f"  Times  : {times_str}")
        print(f"  Combos : {combos_str}\n")
        results[name] = {"dashes": total, "timestamps": timestamps, "combos": [[n, lbl] for n, lbl in combos]}

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUTPUT_DIR) / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
