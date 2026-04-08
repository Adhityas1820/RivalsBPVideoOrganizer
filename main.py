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
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from game_mode_select import is_domination

# ---------------------------------------------------------------------------
# Config — paths
# ---------------------------------------------------------------------------
UNSORTED_DIR   = "unsorted_videos"
SORTED_DIR     = "final"
MAP_MODEL_PATH = "models/map_classifier.pth"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Config — kill counter
# ---------------------------------------------------------------------------
KILL_PROCESS_FPS = 60

KILL_X_START      = 1550
KILL_X_END        = 1875
KILL_ROW_Y_STARTS = [30, 65, 100, 135, 170]
KILL_ROW_H        = 35

WHITE_THRESH      = 200
ROW_WHITE_RATIO   = 0.20
KILL_STABLE_FRAMES = 3

# ---------------------------------------------------------------------------
# Config — dash counter
# ---------------------------------------------------------------------------
DASH_PROCESS_FPS = 60

SLOT2_SEARCH = (1575, 1625, 965, 1000)
SLOT3_SEARCH = (1500, 1550, 965, 1000)
SLOT2_LABEL  = (1575, 1625, 1008, 1050)
SLOT3_LABEL  = (1500, 1550, 1008, 1050)

SLOT_DETECT_FRAMES = 30

RIGHT_CONTOUR_PATH = "reference pictures/slot_x_contour_right.npy"
LEFT_CONTOUR_PATH  = "reference pictures/slot_x_contour_left.npy"

WHITE_THRESH_DASH   = 200
WHITE_RATIO_THRESH  = 1.0
ZOOM_LOW_THRESH     = 0.2
OFF_FRAMES          = 5
DASH_REARM_SECS     = 0.3

LABEL_WHITE_THRESH  = 0.1
LABEL_STABLE_FRAMES = 10
COMBO_GAP_SECS      = 0.9
COMBO_NAMES         = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}

# ---------------------------------------------------------------------------
# Config — map classifier
# ---------------------------------------------------------------------------
FRAME_INTERVAL_SECONDS = 2
IMG_SIZE = 224

DOMINATION_MAPS = {"Birnin TChalla", "Celestial Husk", "Hells Heaven", "Krakoa", "Royal Palace"}

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


def crop_region(frame: np.ndarray, region: tuple) -> np.ndarray:
    x0, x1, y0, y1 = region
    h, w = frame.shape[:2]
    return frame[min(y0, h):min(y1, h), min(x0, w):min(x1, w)]


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"

# ---------------------------------------------------------------------------
# Kill counter
# ---------------------------------------------------------------------------

def row_is_lit(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (gray > WHITE_THRESH).sum() / gray.size >= ROW_WHITE_RATIO


def count_kills(video_path: Path) -> tuple:
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return video_path.name, 0, []

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / KILL_PROCESS_FPS))

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
                1 for y0 in KILL_ROW_Y_STARTS
                if row_is_lit(frame[y0:min(y0+KILL_ROW_H,hf), min(KILL_X_START,wf):min(KILL_X_END,wf)])
            )

            if lit == candidate_count:
                streak += 1
            else:
                candidate_count = lit
                streak          = 1

            if streak == KILL_STABLE_FRAMES:
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

# ---------------------------------------------------------------------------
# Dash counter
# ---------------------------------------------------------------------------

def dash_label_white_ratio(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (gray > WHITE_THRESH_DASH).sum() / max(gray.size, 1)


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
    white_pixels = ((gray > WHITE_THRESH_DASH) & (mask > 0)).sum()
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
    return ((gray_z > WHITE_THRESH_DASH) & outside).sum() / total if total > 0 else 0.0


def load_dash_contours(path_str):
    p = Path(path_str)
    if not p.exists():
        return []
    data = np.load(str(p), allow_pickle=True)
    return list(data)


def count_dashes(video_path: Path) -> tuple:
    """Returns (name, total_dashes, timestamp_strings, combos)."""
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return video_path.name, 0, [], []

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / DASH_PROCESS_FPS))
    rearm_frames   = int(DASH_REARM_SECS * src_fps)

    ratio2_acc = ratio3_acc = 0.0
    sampled = 0
    while sampled < SLOT_DETECT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        ratio2_acc += dash_label_white_ratio(frame, *SLOT2_LABEL)
        ratio3_acc += dash_label_white_ratio(frame, *SLOT3_LABEL)
        sampled += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    is_right      = ratio2_acc >= ratio3_acc
    slot_contours = load_dash_contours(RIGHT_CONTOUR_PATH if is_right else LEFT_CONTOUR_PATH)
    active_label  = SLOT2_LABEL if is_right else SLOT3_LABEL

    if slot_contours:
        rx, ry, rw, rh = cv2.boundingRect(np.concatenate(slot_contours).astype(np.int32))
        sx0, sx1, sy0, sy1 = rx, rx + rw, ry, ry + rh
    else:
        sx0, sx1, sy0, sy1 = SLOT2_SEARCH if is_right else SLOT3_SEARCH

    off_streak = 0
    was_off    = True
    rearm_at   = 0
    total_dashes = 0
    timestamps   = []

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

            lbl_ratio     = dash_label_white_ratio(frame, *active_label)
            label_is_grey = lbl_ratio < LABEL_WHITE_THRESH
            if label_is_grey == label_candidate_grey:
                label_candidate_streak += 1
            else:
                label_candidate_grey   = label_is_grey
                label_candidate_streak = 1
            prev_label_stable_grey = label_stable_grey
            if label_candidate_streak >= LABEL_STABLE_FRAMES:
                label_stable_grey = label_candidate_grey

            if prev_label_stable_grey and not label_stable_grey:
                if combo_count >= 2:
                    combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))
                combo_count         = 0
                last_combo_dash_sec = None

        read_idx += 1

    if combo_count >= 2:
        combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return video_path.name, total_dashes, timestamps, combos

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

def _kill_worker(path_str: str):
    return count_kills(Path(path_str))


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

    # Kills (parallel)
    print("Counting kills...")
    with mp.Pool(processes=num_workers) as pool:
        kill_results = pool.map(_kill_worker, [str(v) for v in videos])
    kills_by_name = {name: total for name, total, _ in kill_results}

    # Dashes (parallel)
    print("Counting dashes...")
    with mp.Pool(processes=num_workers) as pool:
        dash_results = pool.map(_dash_worker, [str(v) for v in videos])
    dashes_by_name = {name: (total, combos) for name, total, _, combos in dash_results}

    # Map classification (sequential, uses GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Classifying maps  (device: {device})...")
    classifier, classes = load_map_classifier(MAP_MODEL_PATH, device)

    print("=" * 60)

    for video_path in videos:
        name = video_path.name

        pil_frames = extract_pil_frames(video_path)
        if not pil_frames:
            print(f"[SKIP] {name} — could not read video.")
            continue

        dom, dom_conf = is_domination(video_path)
        if dom:
            allowed = DOMINATION_MAPS
        else:
            allowed = {c for c in classes if c not in DOMINATION_MAPS}

        map_name, conf = classify_video(pil_frames, classifier, classes, device, allowed=allowed)
        kills          = kills_by_name.get(name, 0)
        dashes, combos = dashes_by_name.get(name, (0, []))

        best_combo = max(combos, key=lambda x: x[0]) if combos else None
        if best_combo:
            new_stem = f"{map_name} - {best_combo[1]} - {dashes} dashes - {kills} kills"
        else:
            new_stem = f"{map_name} - {dashes} dashes - {kills} kills"
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
        print(f"  Combo  : {best_combo[1] if best_combo else 'none'}")
        print(f"  Saved  : {dest_path}\n")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
