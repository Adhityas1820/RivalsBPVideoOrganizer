# -*- coding: utf-8 -*-
"""pipeline.py — Marvel Rivals Classifier processing pipeline.

All heavy lifting (frame extraction, map classification, kill/dash/combo
counting, organising clips) lives here so the UI layer is a thin shell.

The single public entry point is ``run_pipeline(videos, options, emit)``.
``emit(event_type, **fields)`` is a callback used to stream progress back to
whatever frontend is driving the pipeline. Event types match what the old
Tkinter GUI used, so behaviour is identical:

    emit('log',      message=..., tag='')         # log line ('', 'ok', 'warn', 'error')
    emit('stage',    stage=..., status='active'|'done')
    emit('progress', value=<0-100>)
    emit('done',     results=[...])
    emit('error',    message=...)
"""

import shutil
import zipfile
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# torch / torchvision / PIL are imported lazily inside the functions that use
# them — importing them here would add several seconds to app launch before
# the window can even appear.

from video_io import open_video
from dash_counter import count_dashes
from kill_counter import (count_kills, load_contours as load_kill_contours,
                          SLOT1_CONTOUR_PATH as BOARD_SLOT1_CONTOUR_PATH,
                          SLOT2_CONTOUR_PATH as BOARD_SLOT2_CONTOUR_PATH)
from clip_export import trim_clip

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SORTED_DIR       = "final"
MAP_MODEL_PATH   = "models/map_classifier.pth"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Dash detection method: 'contour' (classical OpenCV, default), 'fdnn'
# (neural-net SDSNN detector), or 'hybrid' (run both, keep the higher count).
# Overridden per-run by options['dash_method'] / options['fdnn_threshold'].
DASH_METHOD     = "contour"
FDNN_THRESHOLD  = 0.7   # NMS confidence cutoff for the FDNN detector
FDNN_MODEL_PATH = "models/sdsnn.pt"

# Output mode: 'rename' (default, copy the whole clip like today) or 'clip'
# (trim out just the qualifying dash-combo/kill-streak moments). Overridden
# per-run by options['output_mode'] / ['dash_clip_min'] / ['kill_clip_min'] /
# ['clip_pad_secs'].
OUTPUT_MODE    = "rename"
DASH_CLIP_MIN  = 2     # minimum combo tier to clip: 2=Double .. 5=Penta
KILL_CLIP_MIN  = 1     # minimum kills in a streak to clip
CLIP_PAD_SECS  = 2.0   # buffer seconds added before/after each moment

FRAME_INTERVAL_SECONDS = 0.5
IMG_SIZE = 224
BLACKOUT_BOXES = [
    (25,   600,  900, 1060),
    (1450, 1875, 900, 1065),
    (750,  1175, 970, 1050),
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = None


def _get_transform():
    """Build (and cache) the torchvision preprocessing pipeline on first use."""
    global _transform
    if _transform is None:
        from torchvision import transforms
        _transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return _transform

# Stage metadata shared with the UI.
STAGE_META = [
    ('kills_dashes', 'Kills & Dashes'),
    ('frames',       'Frame Extract'),
    ('classify',     'Map Classify'),
    ('organize',     'Organize'),
]
STAGE_MSG = {
    'kills_dashes': 'Counting kills & dashes...',
    'frames':       'Extracting frames...',
    'classify':     'Classifying maps...',
    'organize':     'Organising clips...',
}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def apply_blackout(frame):
    for x0, x1, y0, y1 in BLACKOUT_BOXES:
        frame[y0:y1, x0:x1] = 0
    return frame


def load_map_classifier(model_path: str, device):
    import torch
    from torchvision import models
    ckpt    = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model   = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes


def extract_pil_frames(video_path: Path) -> list:
    from PIL import Image
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return []
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * FRAME_INTERVAL_SECONDS))
    # Sequential grab() + retrieve() of every step-th frame: same frames as
    # seeking to each sample, but without re-decoding from the previous
    # keyframe for every seek (which made this several times slower).
    idx, frames = 0, []
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            apply_blackout(frame)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return frames


def _kill_worker(args):
    path_str, bc1, bc2 = args
    return count_kills(Path(path_str), bc1, bc2)


def _dash_worker(path_str):
    return count_dashes(Path(path_str))


def model_ready() -> bool:
    return Path(MAP_MODEL_PATH).exists()


def fdnn_model_ready() -> bool:
    """True when the FDNN dash detector can run (checkpoint + backbone present)."""
    try:
        from fdnn import counter as _fdnn_counter
        return _fdnn_counter.model_ready()
    except Exception:
        return False


def _fdnn_dashes(videos, threshold, emit):
    """Run the FDNN dash detector over videos in the main process (it loads a
    torch model, so it isn't farmed out to the mp pool). Returns the same list
    of 5-tuples the contour detector produces. Emits incremental progress."""
    from fdnn.counter import count_dashes_fdnn
    results = []
    n = len(videos)
    for i, v in enumerate(videos):
        try:
            results.append(count_dashes_fdnn(v, threshold))
        except Exception as exc:
            emit('log', message=f'  [{v.name}] FDNN failed: {exc}', tag='warn')
            results.append((v.name, 0, [], [], []))
        emit('progress', value=5 + int(33 * (i + 1) / max(n, 1)))
    return results


def _merge_dash_results(method, contour_results, fdnn_results):
    """Combine the two detectors' outputs per the chosen method. 'hybrid' keeps,
    for each video, whichever detector reported MORE dashes (ties -> contour)."""
    if method == 'contour' or not fdnn_results:
        return contour_results
    if method == 'fdnn' or not contour_results:
        return fdnn_results
    cmap = {r[0]: r for r in contour_results}
    fmap = {r[0]: r for r in fdnn_results}
    merged = []
    for nm, cres in cmap.items():
        fres = fmap.get(nm)
        merged.append(cres if (fres is None or cres[1] >= fres[1]) else fres)
    for nm, fres in fmap.items():       # any fdnn-only names (shouldn't happen)
        if nm not in cmap:
            merged.append(fres)
    return merged


def _unique_dest(stem: str, suffix: str) -> Path:
    dest = Path(SORTED_DIR) / f'{stem}{suffix}'
    c2 = 1
    while dest.exists():
        dest = Path(SORTED_DIR) / f'{stem} ({c2}){suffix}'
        c2 += 1
    return dest


def dash_qualifying_moments(combos, min_tier: int, pad: float) -> list:
    """combos: [(count, label, start_sec, end_sec), ...]. Returns
    [(clip_start, clip_end, label), ...] for combos at or above min_tier."""
    return [(max(0.0, start - pad), end + pad, label)
            for count, label, start, end in combos if count >= min_tier]


def kill_qualifying_moments(streaks, min_kills: int, pad: float) -> list:
    """streaks: [(count, start_sec, end_sec), ...]. Returns
    [(clip_start, clip_end, label), ...] for streaks at or above min_kills."""
    return [(max(0.0, start - pad), end + pad, f'{count}k Streak')
            for count, start, end in streaks if count >= min_kills]


def merge_moments(moments: list) -> list:
    """Merge overlapping/touching (start, end, label) moments into single spans
    so the same footage is never exported as more than one clip (a dash combo
    and a kill streak can land in the same window). Combined labels join with
    ' + '. Returns moments sorted by start time."""
    if not moments:
        return []
    ordered = sorted(moments, key=lambda m: m[0])
    merged = [list(ordered[0])]
    for start, end, label in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
            if label not in last[2].split(' + '):
                last[2] = f'{last[2]} + {label}'
        else:
            merged.append([start, end, label])
    return [tuple(m) for m in merged]


# ---------------------------------------------------------------------------
# Export helpers (used by the UI's save / zip actions)
# ---------------------------------------------------------------------------

def export_clip(filename: str, dst: str) -> bool:
    src = Path(SORTED_DIR) / filename
    if not src.exists():
        return False
    shutil.copy2(str(src), dst)
    return True


def export_zip(filenames, dst: str) -> int:
    files = [Path(SORTED_DIR) / f for f in filenames]
    files = [f for f in files if f.exists()]
    if not files:
        return 0
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, f.name)
    return len(files)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(videos, options, emit):
    """Run the full pipeline.

    videos  : list[Path]
    options : dict with bool keys 'map', 'kills', 'dashes', 'combos'
    emit    : callback(event_type, **fields)
    """
    try:
        videos     = [Path(v) for v in videos]
        run_map    = options.get('map', True)
        run_kills  = options.get('kills', True)
        run_dashes = options.get('dashes', True)
        run_combos = options.get('combos', True)

        output_mode   = (options.get('output_mode') or OUTPUT_MODE).lower()
        dash_clip_min = int(options.get('dash_clip_min', DASH_CLIP_MIN))
        kill_clip_min = int(options.get('kill_clip_min', KILL_CLIP_MIN))
        clip_pad_secs = float(options.get('clip_pad_secs', CLIP_PAD_SECS))
        if output_mode not in ('rename', 'clip'):
            output_mode = 'rename'

        if not videos:
            emit('error', message='No videos selected.')
            return
        if run_map and not Path(MAP_MODEL_PATH).exists():
            emit('error', message=f'Model not found at {MAP_MODEL_PATH}.')
            return

        nw      = max(1, mp.cpu_count() - 1)
        enabled = ' + '.join(x for x, b in [('Map', run_map), ('Kills', run_kills),
                                            ('Dashes', run_dashes), ('Combos', run_combos)] if b) or 'nothing'
        emit('log', message=f'Processing {len(videos)} video(s)  |  workers: {nw}  |  running: {enabled}')

        kills_map  = {}
        dashes_map = {}

        # Pick the dash detection method for this run.
        method   = (options.get('dash_method') or DASH_METHOD).lower()
        fdnn_thr = float(options.get('fdnn_threshold', FDNN_THRESHOLD))
        if method not in ('contour', 'fdnn', 'hybrid'):
            method = 'contour'

        need_dashes = run_dashes or run_combos
        use_contour = need_dashes and method in ('contour', 'hybrid')
        use_fdnn    = need_dashes and method in ('fdnn', 'hybrid')
        if use_fdnn and not fdnn_model_ready():
            emit('log', message='FDNN model unavailable — using contour detector instead.', tag='warn')
            use_fdnn    = False
            use_contour = need_dashes          # fall back so dashes still run

        if run_kills or need_dashes:
            emit('stage', stage='kills_dashes', status='active')
            emit('progress', value=5)
            if need_dashes:
                emit('log', message=f'Dash detector: {method.upper()}'
                                    + (f' (threshold {fdnn_thr:.2f})' if use_fdnn else ''))

            kill_results, contour_results = [], []
            # Kills + contour dashes share the multiprocessing pool.
            if run_kills or use_contour:
                with mp.Pool(processes=nw) as pool:
                    kf = df = None
                    if run_kills:
                        bc1 = load_kill_contours(BOARD_SLOT1_CONTOUR_PATH)
                        bc2 = load_kill_contours(BOARD_SLOT2_CONTOUR_PATH)
                        kf  = pool.map_async(_kill_worker, [(str(v), bc1, bc2) for v in videos])
                    if use_contour:
                        df  = pool.map_async(_dash_worker, [str(v) for v in videos])
                    if kf is not None:
                        kill_results = kf.get()
                    if df is not None:
                        contour_results = df.get()

            # FDNN dashes run in the main process (torch model).
            fdnn_results = _fdnn_dashes(videos, fdnn_thr, emit) if use_fdnn else []

            if run_kills:
                kills_map = {nm: (tot, streaks) for nm, tot, _, streaks, __ in kill_results}
                for nm, tot, _, streaks, __ in kill_results:
                    emit('log', message=f'  [{nm}]  kills {tot}')
            if need_dashes:
                dash_results = _merge_dash_results(method, contour_results, fdnn_results)
                dashes_map = {nm: (tot, cb) for nm, tot, _, cb, __ in dash_results}
                for nm, tot, _, cb, __ in dash_results:
                    cs = ('  (' + ' - '.join(lbl for _, lbl, *_ in cb) + ')') if cb else ''
                    emit('log', message=f'  [{nm}]  dashes {tot}{cs}')

            emit('stage', stage='kills_dashes', status='done')

        emit('progress', value=40)

        map_by_name = {}
        if run_map:
            import torch
            import torch.nn.functional as F
            emit('stage', stage='frames', status='active')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            emit('log', message=f'Device: {device}')
            classifier, classes = load_map_classifier(MAP_MODEL_PATH, device)

            with ThreadPoolExecutor(max_workers=nw) as ex:
                video_data = list(ex.map(extract_pil_frames, videos))

            emit('progress', value=65)
            emit('stage', stage='frames', status='done')

            emit('stage', stage='classify', status='active')
            transform = _get_transform()
            all_tensors, frame_counts, valid_idx = [], [], []
            for i, pil_frames in enumerate(video_data):
                if not pil_frames:
                    frame_counts.append(0)
                    continue
                ts = [transform(pf) for pf in pil_frames]
                all_tensors.extend(ts)
                frame_counts.append(len(ts))
                valid_idx.append(i)

            if all_tensors:
                batch_size = len(all_tensors)
                all_probs  = None
                while batch_size >= 1:
                    try:
                        chunks = []
                        with torch.no_grad():
                            for i in range(0, len(all_tensors), batch_size):
                                mini = torch.stack(all_tensors[i:i+batch_size]).to(device)
                                chunks.append(F.softmax(classifier(mini), dim=1).cpu().numpy())
                        all_probs = np.concatenate(chunks, axis=0)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        batch_size //= 2
                        emit('log', message=f'  OOM — retrying with batch size {batch_size}')
                if all_probs is None:
                    raise RuntimeError('Could not run inference even at batch size 1.')
                ptr = 0
                for i in valid_idx:
                    cnt  = frame_counts[i]
                    avg  = np.mean(all_probs[ptr:ptr + cnt], axis=0)
                    ptr += cnt
                    best = int(np.argmax(avg))
                    map_by_name[videos[i].name] = (classes[best], float(avg[best]))

            emit('progress', value=80)
            emit('stage', stage='classify', status='done')

        emit('progress', value=80)
        emit('stage', stage='organize', status='active')
        Path(SORTED_DIR).mkdir(parents=True, exist_ok=True)
        results = []
        n = len(videos)
        for idx, vp in enumerate(videos):
            nm = vp.name
            map_name, conf = 'Unknown', 0
            if run_map:
                if nm not in map_by_name:
                    emit('log', message=f'  SKIP {nm}')
                    continue
                map_name, conf = map_by_name[nm]
            kills, streaks = kills_map.get(nm, (0, []))
            dashes, combos = dashes_map.get(nm, (0, []))
            combo_str      = (' - '.join(lbl for _, lbl, *_ in combos)) if combos else None

            if output_mode == 'clip':
                moments = (dash_qualifying_moments(combos, dash_clip_min, clip_pad_secs) +
                          kill_qualifying_moments(streaks, kill_clip_min, clip_pad_secs))
                if not moments:
                    emit('log', message=f'  SKIP {nm} (no qualifying moments)')
                    continue
                moments = merge_moments(moments)
                for start, end, label in moments:
                    parts = []
                    if run_map: parts.append(map_name)
                    parts.append(label)
                    if run_dashes: parts.append(f'{dashes}d')
                    if run_kills:  parts.append(f'{kills}k')
                    stem = ' - '.join(parts)

                    dest = _unique_dest(stem, vp.suffix)
                    if not trim_clip(vp, start, end, dest):
                        emit('log', message=f'  FAILED to clip {nm} @ {label}', tag='warn')
                        continue
                    results.append({'original': nm, 'filename': dest.name,
                                    'map': map_name, 'conf': round(conf * 100),
                                    'kills': kills, 'dashes': dashes, 'combos': combo_str,
                                    'clip_label': label})
                    emit('log', message=f'  {dest.name}', tag='ok')
            else:
                parts = []
                if run_map:               parts.append(map_name)
                if run_dashes:            parts.append(f'{dashes}d')
                if run_combos and combo_str: parts.append(combo_str)
                if run_kills:             parts.append(f'{kills}k')
                stem = ' - '.join(parts) if parts else vp.stem

                dest = _unique_dest(stem, vp.suffix)
                shutil.copy2(str(vp), str(dest))
                results.append({'original': nm, 'filename': dest.name,
                                'map': map_name, 'conf': round(conf * 100),
                                'kills': kills, 'dashes': dashes, 'combos': combo_str})
                emit('log', message=f'  {dest.name}', tag='ok')

            emit('progress', value=80 + int(20 * (idx + 1) / n))

        emit('stage', stage='organize', status='done')
        emit('progress', value=100)
        emit('done', results=results)

    except Exception as exc:
        import traceback
        emit('error', message=f'{exc}\n{traceback.format_exc()}')
