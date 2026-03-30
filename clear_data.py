"""
clear_data.py
-------------
Clears extracted frames so you can re-run process_data.py and train_game_mode.py
from scratch.

  dataset_frames/ — deletes all image files, keeps folder structure
  hud_frames/     — deleted entirely (train_game_mode.py recreates it)

Usage:
    python clear_data.py
"""

import shutil
from pathlib import Path

DATASET_FRAMES_DIR = "dataset_frames"
HUD_FRAMES_DIR     = "hud_frames"


def main():
    # dataset_frames — delete files, keep folders
    frames_root = Path(DATASET_FRAMES_DIR)
    if frames_root.exists():
        deleted = 0
        for f in frames_root.rglob("*"):
            if f.is_file():
                f.unlink()
                deleted += 1
        print(f"dataset_frames : {deleted} files deleted, folders preserved.")
    else:
        print(f"dataset_frames : not found, skipping.")

    # hud_frames — delete entirely
    hud_root = Path(HUD_FRAMES_DIR)
    if hud_root.exists():
        shutil.rmtree(hud_root)
        print(f"hud_frames     : deleted entirely.")
    else:
        print(f"hud_frames     : not found, skipping.")


if __name__ == "__main__":
    main()
