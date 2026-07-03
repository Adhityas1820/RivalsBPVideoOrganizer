"""video_io.py — tiny shared helpers for the video-processing modules.

These were previously duplicated across pipeline.py, dash_counter.py,
kill_counter.py and fdnn/features.py.
"""

import shutil
import tempfile
from pathlib import Path

import cv2


def open_video(video_path: Path):
    """Open a video, falling back to a temp copy for paths OpenCV can't read
    directly (e.g. some unicode paths). Returns (cap, tmp_path_or_None); the
    caller must unlink tmp_path after releasing the capture."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    tmp = tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    return cv2.VideoCapture(tmp.name), tmp.name


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"
