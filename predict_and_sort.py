"""
predict_and_sort.py
-------------------
Classifies each video in unsorted_videos/ by map using the ResNet18
map classifier and copies each video into sorted_videos/<MapName>/.

Usage:
    python predict_and_sort.py
"""

import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import models, transforms
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAP_MODEL_PATH         = "models/map_classifier.pth"
UNSORTED_DIR           = "unsorted_videos"
SORTED_DIR             = "sorted_videos"
FRAME_INTERVAL_SECONDS = 2
IMG_SIZE               = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# ---------------------------------------------------------------------------

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_map_classifier(model_path: str, device):
    ckpt    = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model   = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def classify_video(pil_frames: list, classifier, classes: list, device) -> tuple:
    """Averages softmax probabilities across sampled frames. Returns (class_name, confidence)."""
    all_probs = []
    for pil in pil_frames:
        tensor = _transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(classifier(tensor), dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    avg      = np.mean(all_probs, axis=0)
    best_idx = int(np.argmax(avg))
    return classes[best_idx], float(avg[best_idx])


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    tmp = tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    return cv2.VideoCapture(tmp.name), tmp.name


def extract_pil_frames(video_path: Path, interval_seconds: float) -> list:
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return []

    fps            = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps * interval_seconds))
    frame_idx      = 0
    frames         = []

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += frame_interval

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return frames


def main():
    unsorted_path = Path(UNSORTED_DIR)
    unsorted_path.mkdir(parents=True, exist_ok=True)

    videos = sorted([
        f for f in unsorted_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])
    if not videos:
        print(f"No videos in '{UNSORTED_DIR}/'. Nothing to do.")
        return

    if not Path(MAP_MODEL_PATH).exists():
        print(f"[ERROR] Model not found at '{MAP_MODEL_PATH}'. Run train_model.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Videos to classify: {len(videos)}")
    print(f"Device: {device}\n")

    print("Loading map classifier...")
    classifier, classes = load_map_classifier(MAP_MODEL_PATH, device)
    print(f"  Classes ({len(classes)}): {classes}")
    print("=" * 60)

    results = []

    for video_path in videos:
        print(f"\n[{video_path.name}]")

        pil_frames = extract_pil_frames(video_path, FRAME_INTERVAL_SECONDS)
        if not pil_frames:
            print("  Could not read video — skipping.")
            continue
        print(f"  Frames : {len(pil_frames)}")

        map_name, conf = classify_video(pil_frames, classifier, classes, device)
        print(f"  Map    : {map_name}  ({conf*100:.1f}%)")

        dest_dir  = Path(SORTED_DIR) / map_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_name  = f"[{map_name} - {conf*100:.0f}%] {video_path.name}"
        dest_path = dest_dir / new_name
        if dest_path.exists():
            stem, sfx, n = dest_path.stem, dest_path.suffix, 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{n}{sfx}"
                n += 1

        shutil.copy2(str(video_path), str(dest_path))
        print(f"  Saved  : {dest_path}")
        results.append((video_path.name, map_name, conf))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, map_name, conf in results:
        print(f"  {name:<45} -> [{map_name}]  ({conf*100:.0f}%)")
    print(f"\n{len(results)}/{len(videos)} video(s) sorted into '{SORTED_DIR}/'.")


if __name__ == "__main__":
    main()
