"""
kill_counter.py
---------------
Counts kills in Marvel Rivals gameplay clips by monitoring 5 stacked rows
of the kill feed region. Each row is considered "lit" when its white pixel
ratio exceeds ROW_WHITE_RATIO. Kills are counted when the total number of
lit rows increases (stable for STABLE_FRAMES consecutive processed frames).

Handles simultaneous kills (multiple rows light up at once) and ult kill
animations (lightning bridges blobs but keeps white ratio high).

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

X_START      = 1550
X_END        = 1875
ROW_Y_STARTS = [30, 65, 100, 135, 170]
ROW_H        = 35

WHITE_THRESH    = 200
ROW_WHITE_RATIO = 0.20

STABLE_FRAMES = 3

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


def row_is_lit(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (gray > WHITE_THRESH).sum() / gray.size >= ROW_WHITE_RATIO


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def count_kills(video_path: Path) -> tuple:
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return video_path.name, 0, []

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / PROCESS_FPS))

    stable_lit_count = 0
    candidate_count  = 0
    streak           = 0
    total_kills      = 0
    timestamps       = []
    read_idx         = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx % frame_interval == 0:
            hf, wf = frame.shape[:2]

            lit = sum(
                1 for y0 in ROW_Y_STARTS
                if row_is_lit(frame[y0:min(y0+ROW_H,hf), min(X_START,wf):min(X_END,wf)])
            )

            if lit == candidate_count:
                streak += 1
            else:
                candidate_count = lit
                streak          = 1

            if streak == STABLE_FRAMES:
                if candidate_count > stable_lit_count:
                    new_kills    = candidate_count - stable_lit_count
                    total_kills += new_kills
                    t = fmt_timestamp(read_idx / src_fps)
                    for _ in range(new_kills):
                        timestamps.append(t)
                stable_lit_count = candidate_count

        read_idx += 1

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return video_path.name, total_kills, timestamps


def _worker(video_path_str: str) -> tuple:
    return count_kills(Path(video_path_str))


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
