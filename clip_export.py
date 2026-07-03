# -*- coding: utf-8 -*-
"""clip_export.py — trims a highlight moment out of a source video via ffmpeg.

Used by pipeline.py's Clip & Rename output mode. Stream-copies (no re-encode)
for speed and lossless quality; the bundled ffmpeg binary comes from the
imageio-ffmpeg package so no separate install/PATH entry is required.

Caveat: with `-c copy`, `-ss` snaps to the nearest preceding keyframe, so the
actual clip start may land up to one GOP length before the requested start.
This is an accepted trade-off for speed — the configurable padding already
adds a buffer that absorbs most of the drift.
"""

import subprocess
import sys
from pathlib import Path

_ffmpeg_exe = None
_NO_WINDOW  = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def ffmpeg_path() -> str:
    global _ffmpeg_exe
    if _ffmpeg_exe is None:
        import imageio_ffmpeg
        _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    return _ffmpeg_exe


def trim_clip(src: Path, start_sec: float, end_sec: float, dst: Path) -> bool:
    """Cut [start_sec, end_sec] out of src into dst via ffmpeg stream-copy.
    Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(), '-y',
        '-ss', str(max(0.0, start_sec)),
        '-to', str(end_sec),
        '-i', str(src),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if result.returncode != 0:
        print(f"[clip_export] ffmpeg failed for {src.name} -> {dst.name}: {result.stderr[-500:]}")
        return False
    return True
