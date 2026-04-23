"""
dash_counter.py
---------------
Counts dashes in Marvel Rivals gameplay clips using contour-based white
detection + zoom exclusion.

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
SLOT2_LABEL  = (1575, 1625, 1030, 1050)
SLOT3_LABEL  = (1500, 1550, 1030, 1050)
SLOT_DETECT_FRAMES = 240

RIGHT_CONTOUR_PATH = "models/slot_x_contour_right.npy"
LEFT_CONTOUR_PATH  = "models/slot_x_contour_left.npy"

WHITE_THRESH       = 200
WHITE_RATIO_THRESH = 0.95
LABEL_GREY_THRESH  = 110
ZOOM_LOW_THRESH    = 0.5
OFF_FRAMES         = 3
DASH_REARM_SECS    = 0.3

COMBO_WINDOW_PER_DASH = 0.3   # window = n * 450ms from first dash (900ms=Double, 1350=Triple, 1800=Quad, 2250=Penta)
COMBO_NAMES           = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}

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


def count_label_contours(frame, x0, x1, y0, y1) -> int:
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, LABEL_GREY_THRESH, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts)


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
    """Returns (video_name, total_dashes, timestamp_strings, combos, dash_secs)"""
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return video_path.name, 0, [], [], []

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / PROCESS_FPS))
    rearm_frames   = int(DASH_REARM_SECS * src_fps)

    # Detect which slot has the LSHIFT label by counting contours in left label box
    left_cnts_acc = 0
    sampled = 0
    for fi in range(0, SLOT_DETECT_FRAMES, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        left_cnts_acc += count_label_contours(frame, *SLOT3_LABEL)
        sampled += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    avg_cnts = left_cnts_acc / max(sampled, 1)
    is_right = avg_cnts > 1
    print(f"  [{video_path.name}] Dash slot: {'RIGHT (slot2)' if is_right else 'LEFT (slot3)'}  (avg contours: {avg_cnts:.2f})")
    slot_contours = load_contours(RIGHT_CONTOUR_PATH if is_right else LEFT_CONTOUR_PATH)

    if slot_contours:
        rx, ry, rw, rh = cv2.boundingRect(np.concatenate(slot_contours).astype(np.int32))
        sx0, sx1, sy0, sy1 = rx, rx + rw, ry, ry + rh
    else:
        sx0, sx1, sy0, sy1 = SLOT2_SEARCH if is_right else SLOT3_SEARCH

    off_streak   = 0
    was_off      = True
    rearm_at     = 0
    total_dashes = 0
    timestamps   = []
    dash_secs    = []

    combo_count      = 0
    combo_start_sec  = None
    combos           = []

    read_idx     = 0

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
                dash_secs.append(t_sec)
                rearm_at = read_idx + rearm_frames
                was_off  = False

                if combo_start_sec is None:
                    combo_start_sec = t_sec
                    combo_count     = 1
                else:
                    new_count = combo_count + 1
                    if (t_sec - combo_start_sec) <= new_count * COMBO_WINDOW_PER_DASH:
                        combo_count = new_count
                    else:
                        if combo_count >= 2:
                            combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))
                        combo_start_sec = t_sec
                        combo_count     = 1

            if not white_state:
                off_streak += 1
                if off_streak >= OFF_FRAMES:
                    was_off = True
            else:
                off_streak = 0

        read_idx += 1

    if combo_count >= 2:
        combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return video_path.name, total_dashes, timestamps, combos, dash_secs


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

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Write timestamps text file
    txt_file = Path(OUTPUT_DIR) / "dash_timestamps.txt"
    results  = {}
    txt_lines = []

    for name, total, timestamps, combos, dash_secs in all_results:
        times_str  = ", ".join(timestamps) if timestamps else "none"
        combos_str = ", ".join(f"{lbl} ({n})" for n, lbl in combos) if combos else "none"
        print(f"[{name}]")
        print(f"  Dashes : {total}")
        print(f"  Times  : {times_str}")
        print(f"  Combos : {combos_str}\n")
        results[name] = {"dashes": total, "timestamps": timestamps, "combos": [[n, lbl] for n, lbl in combos]}

        txt_lines.append(f"=== {name} ===")
        if dash_secs:
            for i, (t, ts) in enumerate(zip(dash_secs, timestamps)):
                if i == 0:
                    txt_lines.append(f"  dash {i+1:>2}: {ts}")
                else:
                    delta = t - dash_secs[i - 1]
                    txt_lines.append(f"  dash {i+1:>2}: {ts}  (delta: {delta:.3f}s)")
        else:
            txt_lines.append("  no dashes detected")
        txt_lines.append("")

    with open(txt_file, "w") as f:
        f.write("\n".join(txt_lines))

    out_file = Path(OUTPUT_DIR) / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Timestamps : {txt_file}")
    print(f"JSON       : {out_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
