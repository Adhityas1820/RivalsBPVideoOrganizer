"""
main.py
-------
Full pipeline for Marvel Rivals gameplay clips.

For every video in unsorted_videos/:
  1. Count kills   (white blob detection in kill feed)
  2. Count dashes  (yellow icon detection in dash slot)
  3. Classify map  (ResNet18 map classifier)
  4. Copy to final/<MapName>/<MapName> - X kills - Y dashes.<ext>

Usage:
    python main.py
"""

import shutil
import tempfile
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from game_mode_select import is_domination
from dash_counter import count_dashes
from kill_counter import (count_kills, load_contours as load_kill_contours,
                          SLOT1_CONTOUR_PATH as BOARD_SLOT1_CONTOUR_PATH,
                          SLOT2_CONTOUR_PATH as BOARD_SLOT2_CONTOUR_PATH)

# ---------------------------------------------------------------------------
# Config — paths
# ---------------------------------------------------------------------------
UNSORTED_DIR   = "unsorted_videos"
SORTED_DIR     = "final"
MAP_MODEL_PATH = "models/map_classifier.pth"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Config — map classifier
# ---------------------------------------------------------------------------
FRAME_INTERVAL_SECONDS = 2
IMG_SIZE = 224

DOMINATION_MAPS = {"Birnin TChalla", "Celestial Husk", "Hells Heaven", "Krakoa", "Lower Manhattan", "Royal Palace"}

BLACKOUT_BOXES = [
    (25,   600,  900, 1060),
    (1450, 1875, 900, 1065),
    (750,  1175, 970, 1050),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def apply_blackout(frame):
    for x0, x1, y0, y1 in BLACKOUT_BOXES:
        frame[y0:y1, x0:x1] = 0
    return frame


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    tmp = tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    return cv2.VideoCapture(tmp.name), tmp.name


# ---------------------------------------------------------------------------
# Map classifier
# ---------------------------------------------------------------------------

def load_map_classifier(model_path: str, device):
    ckpt    = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model   = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def extract_pil_frames(video_path: Path) -> list:
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return []

    fps            = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps * FRAME_INTERVAL_SECONDS))
    frame_idx      = 0
    frames         = []

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        apply_blackout(frame)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += frame_interval

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return frames


def classify_video(pil_frames: list, classifier, classes: list, device,
                   allowed: set = None) -> tuple:
    all_probs = []
    for pil in pil_frames:
        tensor = _transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(classifier(tensor), dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    avg = np.mean(all_probs, axis=0)
    if allowed:
        mask = np.array([c in allowed for c in classes], dtype=float)
        avg  = avg * mask
    best_idx = int(np.argmax(avg))
    return classes[best_idx], float(avg[best_idx])

# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------

def _kill_worker(args: tuple):
    path_str, bc1, bc2 = args
    return count_kills(Path(path_str), bc1, bc2)


def _dash_worker(path_str: str):
    return count_dashes(Path(path_str))

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    unsorted_path = Path(UNSORTED_DIR)
    videos = sorted([
        f for f in unsorted_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not videos:
        print(f"No videos found in '{UNSORTED_DIR}/'. Nothing to do.")
        return

    if not Path(MAP_MODEL_PATH).exists():
        print(f"[ERROR] Model not found at '{MAP_MODEL_PATH}'. Run train_model.py first.")
        return

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Processing {len(videos)} video(s)  |  workers: {num_workers}\n")
    print("=" * 60)

    # ── Phase 1: kills + dashes in parallel (same pool, overlapping) ──────────
    board_contours1 = load_kill_contours(BOARD_SLOT1_CONTOUR_PATH)
    board_contours2 = load_kill_contours(BOARD_SLOT2_CONTOUR_PATH)
    print(f"Scoreboard slot 1 : {len(board_contours1)} contour(s)")
    print(f"Scoreboard slot 2 : {len(board_contours2)} contour(s)")
    print("Counting kills + dashes (parallel)...")

    kill_args = [(str(v), board_contours1, board_contours2) for v in videos]
    dash_args = [str(v) for v in videos]

    with mp.Pool(processes=num_workers) as pool:
        kill_future = pool.map_async(_kill_worker, kill_args)
        dash_future = pool.map_async(_dash_worker, dash_args)
        kill_results = kill_future.get()
        dash_results = dash_future.get()

    kills_by_name  = {name: total for name, total, _ in kill_results}
    dashes_by_name = {name: (total, combos) for name, total, _, combos, __ in dash_results}

    # ── Phase 2: frame extraction + domination detection (parallel I/O) ───────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting frames + detecting game mode (parallel)  [device: {device}]...")
    classifier, classes = load_map_classifier(MAP_MODEL_PATH, device)

    def extract_video_data(video_path):
        frames = extract_pil_frames(video_path)
        dom, _  = is_domination(video_path)
        return frames, dom

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        video_data = list(ex.map(extract_video_data, videos))

    # ── Phase 3: batched GPU inference ────────────────────────────────────────
    print("Running batched map classification...")
    all_tensors   = []
    frame_counts  = []
    valid_indices = []   # indices into videos[] that had frames

    for i, (pil_frames, _) in enumerate(video_data):
        if not pil_frames:
            frame_counts.append(0)
            continue
        tensors = [_transform(f) for f in pil_frames]
        all_tensors.extend(tensors)
        frame_counts.append(len(tensors))
        valid_indices.append(i)

    map_by_name = {}
    if all_tensors:
        batch = torch.stack(all_tensors).to(device)
        with torch.no_grad():
            all_probs = F.softmax(classifier(batch), dim=1).cpu().numpy()

        ptr = 0
        for i in valid_indices:
            count      = frame_counts[i]
            avg        = np.mean(all_probs[ptr:ptr + count], axis=0)
            ptr       += count
            _, dom     = video_data[i]
            allowed    = DOMINATION_MAPS if dom else {c for c in classes if c not in DOMINATION_MAPS}
            mask       = np.array([c in allowed for c in classes], dtype=float)
            avg        = avg * mask
            best_idx   = int(np.argmax(avg))
            map_by_name[videos[i].name] = (classes[best_idx], float(avg[best_idx]))

    # ── Phase 4: copy files ───────────────────────────────────────────────────
    print("=" * 60)

    for video_path in videos:
        name = video_path.name

        if name not in map_by_name:
            print(f"[SKIP] {name} — could not read video.")
            continue

        map_name, conf = map_by_name[name]
        kills          = kills_by_name.get(name, 0)
        dashes, combos = dashes_by_name.get(name, (0, []))

        if combos:
            combo_str = " - ".join(lbl for _, lbl in combos)
            new_stem  = f"{map_name} - {dashes}d - {combo_str} - {kills}k"
        else:
            new_stem  = f"{map_name} - {dashes}d - {kills}k"

        dest_dir  = Path(SORTED_DIR)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{new_stem}{video_path.suffix}"

        if dest_path.exists():
            n = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{new_stem} ({n}){video_path.suffix}"
                n += 1

        shutil.copy2(str(video_path), str(dest_path))
        print(f"[{name}]")
        print(f"  Map    : {map_name}  ({conf*100:.0f}%)")
        print(f"  Kills  : {kills}")
        print(f"  Dashes : {dashes}")
        print(f"  Combos : {combo_str if combos else 'none'}")
        print(f"  Saved  : {dest_path}\n")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
