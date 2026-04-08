"""
game_mode_selectmk3p3.py
------------------------
Step 1: Locate the HUD by finding stable pixels across frames.
Pixels that change a lot = gameplay/background (eliminated).
Pixels that barely change = HUD elements (kept).

Outputs one image per video to test/{video_name}/hud_stability.jpg showing:
  - Left panel:  a sampled frame
  - Right panel: stability mask (white = stable/HUD, black = changing/background)

Usage:
    python game_mode_selectmk3p3.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_FRAMES  = 200      # frames to sample per video
STABLE_THRESH  = 15      # per-pixel grayscale std below this = stable (HUD)
DEBUG_DIR      = "test"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# ---------------------------------------------------------------------------


def analyze_stability(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open {video_path.name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total_frames // SAMPLE_FRAMES)

    frames_bgr = []
    first_frame = None
    sampled = 0
    read_idx = 0

    while sampled < SAMPLE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % step == 0:
            frames_bgr.append(frame.astype(np.float32))
            if first_frame is None:
                first_frame = frame.copy()
            sampled += 1
        read_idx += 1

    cap.release()

    if len(frames_bgr) < 2:
        print(f"  Not enough frames for {video_path.name}")
        return

    # Per-pixel, per-channel std across all sampled frames
    stack = np.stack(frames_bgr, axis=0)           # (N, H, W, 3)
    std_per_channel = stack.std(axis=0)            # (H, W, 3)
    std_map = std_per_channel.max(axis=2)          # (H, W) — worst-case channel change

    # Stable mask: low std = HUD
    stable_mask = (std_map < STABLE_THRESH).astype(np.uint8) * 255

    # Build side-by-side output image
    h, w = first_frame.shape[:2]
    std_vis = std_map.astype(np.uint8)
    std_vis = cv2.applyColorMap(
        cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )  # heatmap: blue=stable, red=changing

    # Stable mask as BGR
    stable_bgr = cv2.cvtColor(stable_mask, cv2.COLOR_GRAY2BGR)

    # Overlay stable mask as green tint on the original frame
    overlay = first_frame.copy()
    green_layer = np.zeros_like(overlay)
    green_layer[:, :, 1] = 255  # green channel
    mask3 = np.stack([stable_mask, stable_mask, stable_mask], axis=2) > 0
    overlay[mask3] = (overlay[mask3].astype(np.int32) // 2 + 128).clip(0, 255)
    overlay[:, :, 1] = np.where(stable_mask > 0,
                                 np.clip(overlay[:, :, 1].astype(np.int32) + 80, 0, 255),
                                 overlay[:, :, 1])

    # Add labels
    cv2.putText(overlay, "Original + HUD (green=stable)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(std_vis, f"Std heatmap (blue=stable, thresh={STABLE_THRESH})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(stable_bgr, "Stable mask (white=HUD candidate)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Stack: top row = overlay | heatmap, bottom = stable mask (centered)
    top = np.hstack([overlay, std_vis])
    bottom_pad = np.zeros_like(top)
    bottom_pad[:, w//2: w//2 + w] = stable_bgr
    cv2.putText(bottom_pad, "^ stable mask", (w//2 + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    out_img = np.vstack([top, bottom_pad])

    out_dir = Path(DEBUG_DIR) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "hud_stability.jpg")
    cv2.imwrite(out_path, out_img)
    print(f"  Saved: {out_path}  (stable pixels: {(stable_mask > 0).sum():,} / {h*w:,}  = {(stable_mask>0).mean()*100:.1f}%)")


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
        print(f"\n{v.name}")
        analyze_stability(v)


if __name__ == "__main__":
    main()
