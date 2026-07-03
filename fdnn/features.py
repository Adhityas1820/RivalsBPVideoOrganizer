"""features.py — spatial front-end: turn each frame into a 512-d CNN vector.

Adapted from C:/Adhitya/Coding/test/FDNN/dash_code/features.py, trimmed to the
OpenCV decode path only (no decord / torchcodec / tmpfs) so it is self-contained
and packages cleanly into the .exe. A frozen ImageNet ResNet18 (fc head stripped)
reads every frame and emits a 512-d global-avg-pool vector; the SDSNN then models
the ~450 ms dash pattern over that sequence.

The backbone weights are loaded from ``models/resnet18_imagenet.pth`` when present
(bundled for offline use); otherwise torchvision downloads the ImageNet weights.
Output shape: [n_frames, FEAT_DIM], one row per decoded frame, in order.
"""

from pathlib import Path

import cv2
import numpy as np

from video_io import open_video as _open_video
from . import config as C

# Bundled ImageNet ResNet18 weights (so the .exe works offline). Resolved
# relative to the project root, which holds models/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKBONE_WEIGHTS = _PROJECT_ROOT / "models" / "resnet18_imagenet.pth"

N_FEATURES = C.FEAT_DIM

_backbone = None
_device   = None
_mean     = None
_std      = None


def _get_backbone():
    global _backbone, _device, _mean, _std
    if _backbone is not None:
        return _backbone, _device

    import torch
    import torch.nn as nn
    from torchvision import models

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _BACKBONE_WEIGHTS.exists():
        net = models.resnet18(weights=None)
        state = torch.load(str(_BACKBONE_WEIGHTS), map_location="cpu")
        net.load_state_dict(state)
    else:
        # Fallback: let torchvision fetch the ImageNet weights (needs internet
        # the first time). Bundling models/resnet18_imagenet.pth avoids this.
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    net.fc = nn.Identity()                       # expose 512-d avgpool features
    net.eval().to(_device)
    _backbone = net
    _mean = torch.tensor(C.IMAGENET_MEAN, device=_device).view(1, 3, 1, 1)
    _std  = torch.tensor(C.IMAGENET_STD,  device=_device).view(1, 3, 1, 1)
    return _backbone, _device


def _preprocess(frames_bgr):
    """list of HxWx3 uint8 BGR -> normalized NCHW float tensor on _device."""
    import torch
    arr = np.stack([cv2.resize(f, (C.IMG_SIZE, C.IMG_SIZE)) for f in frames_bgr])
    arr = np.ascontiguousarray(arr[:, :, :, ::-1])       # BGR -> RGB
    t = torch.from_numpy(arr).to(_device).float().div_(255.0)
    t = t.permute(0, 3, 1, 2)                            # NHWC -> NCHW
    return (t - _mean) / _std


def extract_cnn_features(video_path) -> np.ndarray:
    """Decode every frame in order and return float32 [n_frames, FEAT_DIM]."""
    import torch
    net, _ = _get_backbone()
    cap, tmp_path = _open_video(video_path)
    if not cap.isOpened():
        cap.release()
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    feats, batch = [], []

    def flush():
        if batch:
            out = net(_preprocess(batch)).cpu().numpy().astype(np.float32)
            feats.append(out)
            batch.clear()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)
            if len(batch) >= C.FEATURE_BATCH:
                flush()
        flush()

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return np.concatenate(feats) if feats else np.zeros((0, N_FEATURES), dtype=np.float32)
