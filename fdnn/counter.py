"""counter.py — FDNN dash counter with the same interface as dash_counter.

``count_dashes_fdnn(video_path, threshold=None)`` runs the full FDNN pipeline
(ResNet18 features -> SDSNN completion probability -> 1-D NMS) and returns the
exact same 5-tuple the classical contour detector returns:

    (video_name, total_dashes, timestamp_strings, combos, dash_secs)

so the pipeline can swap one for the other. Combos are grouped with the same
time-window rule as dash_counter, so a combo means the same thing either way.
"""

from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import features as feat
from . import peaks as pk
from .dnn import DNN

# Trained SDSNN checkpoint (bundled in models/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = _PROJECT_ROOT / "models" / "sdsnn.pt"

_model = None
_ckpt  = None
_device = None


def model_ready() -> bool:
    return CKPT_PATH.exists() and feat._BACKBONE_WEIGHTS.exists()


def load_model():
    """Load (and cache) the SDSNN checkpoint + rebuild its architecture."""
    global _model, _ckpt, _device
    if _model is not None:
        return _model, _ckpt, _device
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(str(CKPT_PATH), map_location=_device)
    model = DNN(in_dim=ck.get("in_dim", C.FEAT_DIM),
                hidden=ck.get("hidden", 64),
                layers=ck.get("layers", 4)).to(_device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    _model, _ckpt = model, ck
    return _model, _ckpt, _device


def _probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = (cap.get(cv2.CAP_PROP_FPS) or C.FPS) if cap.isOpened() else C.FPS
    cap.release()
    return float(fps) or C.FPS


def _fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def combos_from_secs(dash_secs):
    """Group dash start-seconds into combos with the SAME widening time window
    as dash_counter: each new dash joins the run if it lands within
    0.45*(n-1) + 0.275 s of the run's start. Returns [(count, label), ...]."""
    combos = []
    combo_count = 0
    combo_start = None
    for t_sec in dash_secs:
        if combo_start is None:
            combo_start = t_sec
            combo_count = 1
        else:
            new_count = combo_count + 1
            if (t_sec - combo_start) <= 0.45 * (new_count - 1) + 0.275:
                combo_count = new_count
            else:
                if combo_count >= 2:
                    combos.append((combo_count, C.COMBO_NAMES.get(combo_count, f"{combo_count}x")))
                combo_start = t_sec
                combo_count = 1
    if combo_count >= 2:
        combos.append((combo_count, C.COMBO_NAMES.get(combo_count, f"{combo_count}x")))
    return combos


def count_dashes_fdnn(video_path, threshold=None):
    """Returns (video_name, total_dashes, timestamps, combos, dash_secs)."""
    import torch
    video_path = Path(video_path)
    model, ck, device = load_model()
    thr = threshold if threshold is not None else ck.get("threshold", C.PEAK_THRESHOLD)
    min_dist = ck.get("peak_min_dist", C.PEAK_MIN_DIST)

    features = feat.extract_cnn_features(video_path)
    if features.shape[0] == 0:
        return video_path.name, 0, [], [], []

    with torch.no_grad():
        x = torch.from_numpy(features[None]).to(device)        # [1, T, F]
        prob = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()  # [T]

    peak_frames = pk.nms_peaks(prob, thr, min_dist)
    fps = _probe_fps(video_path)
    dash_secs  = [f / fps for f in peak_frames]
    timestamps = [_fmt_timestamp(s) for s in dash_secs]
    combos     = combos_from_secs(dash_secs)
    return video_path.name, len(peak_frames), timestamps, combos, dash_secs
