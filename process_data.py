"""
process_data.py
---------------
Extracts frames from every video in dataset_videos/ and saves them
into dataset_frames/<MapName>/.  No train/val/test split here.

All frames are downscaled to SD (480p) before saving — even if the
source video is 1080p or 4K.

Usage:
    python process_data.py

Input:
    dataset_videos/
    ├── Map1/
    │   ├── game1.mp4
    │   └── game2.mp4
    └── Map2/
        └── game1.mp4

Output:
    dataset_frames/
    ├── Map1/
    │   ├── frame_00000.jpg
    │   └── ...
    └── Map2/
        └── ...
"""

import shutil
import tempfile
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_VIDEOS_DIR     = "dataset_videos"
DATASET_FRAMES_DIR     = "dataset_frames"
FRAME_INTERVAL_SECONDS = 0.5     # one frame every N seconds
SD_HEIGHT              = 480   # downscale target height (480p)
JPEG_QUALITY           = 90    # JPEG quality 0-100

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# HUD regions to black out (1920x1080 coordinates) — keeps CNN focused on map visuals
BLACKOUT_BOXES = [
    (25,   600,  900, 1060),   # slot / ability area
    (1450, 1875, 900, 1065),   # label / LSHIFT text area
    (750,  1175, 970, 1050),   # fill sub-box
]
# ---------------------------------------------------------------------------


def apply_blackout(frame):
    for x0, x1, y0, y1 in BLACKOUT_BOXES:
        frame[y0:y1, x0:x1] = 0
    return frame


def downscale_to_sd(frame, target_height: int = SD_HEIGHT):
    """Resize frame to target_height, preserving aspect ratio. No-op if already small enough."""
    h, w = frame.shape[:2]
    if h <= target_height:
        return frame
    scale = target_height / h
    new_w = int(w * scale)
    new_w += new_w % 2          # ensure even width
    return cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)


def open_video(video_path: Path):
    """
    Open a VideoCapture, working around OpenCV's Windows bug where paths
    containing special characters (apostrophes, accents, etc.) silently fail.
    Falls back to copying the file to a plain-ASCII temp path.
    Returns (cap, tmp_path_or_None).  Caller must delete tmp_path if not None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None

    # First attempt failed — copy to a temp file with a safe ASCII name
    suffix  = video_path.suffix
    tmp     = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    cap = cv2.VideoCapture(tmp.name)
    return cap, tmp.name


def extract_frames(video_path: Path, output_dir: Path, interval_seconds: float, start_idx: int = 0) -> int:
    """
    Extract frames from video_path at interval_seconds spacing, downscaled to SD.
    Frames are numbered starting from start_idx to avoid collisions across videos.
    Returns the number of frames saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        print(f"    [WARNING] Could not open: {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      {src_w}x{src_h}")

    frame_interval  = max(1, int(fps * interval_seconds))
    frame_idx       = 0
    saved_count     = 0
    encode_params   = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            apply_blackout(frame)
            out_path = output_dir / f"frame_{start_idx + saved_count:05d}.jpg"
            cv2.imwrite(str(out_path), frame, encode_params)
            saved_count += 1

        frame_idx += 1

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return saved_count


def main():
    videos_root = Path(DATASET_VIDEOS_DIR)
    frames_root = Path(DATASET_FRAMES_DIR)

    if not videos_root.exists():
        print(f"[ERROR] '{DATASET_VIDEOS_DIR}/' not found.")
        print("Use download_videos.py to populate it, or add videos manually.")
        return

    # Collect all map dirs — supports both flat and nested (Mode/Map/) layouts
    map_dirs = []
    for entry in sorted(videos_root.iterdir()):
        if not entry.is_dir():
            continue
        # Check if this is a game-mode folder (contains subdirs) or a map folder directly
        subdirs = [d for d in entry.iterdir() if d.is_dir()]
        if subdirs:
            # It's a game-mode folder — each subdir is a map
            map_dirs.extend(sorted(subdirs))
        else:
            # It's a flat map folder
            map_dirs.append(entry)

    if not map_dirs:
        print(f"[ERROR] No map folders found inside '{DATASET_VIDEOS_DIR}/'.")
        return

    print(f"Found {len(map_dirs)} map(s): {[d.name for d in map_dirs]}\n")

    total_videos = 0
    total_frames = 0

    for map_dir in map_dirs:
        map_name  = map_dir.name
        out_dir   = frames_root / map_name
        videos    = sorted([
            f for f in map_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])

        if not videos:
            print(f"[WARNING] No videos in '{map_name}/', skipping.\n")
            continue

        print(f"--- {map_name} ({len(videos)} video(s)) ---")

        frame_cursor = 0
        for vid in videos:
            print(f"  {vid.name}")
            n = extract_frames(vid, out_dir, FRAME_INTERVAL_SECONDS, start_idx=frame_cursor)
            print(f"      -> {n} frames saved")
            frame_cursor  += n
            total_frames  += n

        print(f"  Total for {map_name}: {frame_cursor} frames\n")
        total_videos += len(videos)

    print("=" * 50)
    print(f"Done.  Videos processed : {total_videos}")
    print(f"       Total frames saved: {total_frames}")
    print(f"       Output directory  : {DATASET_FRAMES_DIR}/")
    print(f"\nNext step: run  python train_model.py")


if __name__ == "__main__":
    main()
