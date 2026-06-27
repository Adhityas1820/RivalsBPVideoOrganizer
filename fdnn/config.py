"""config.py — constants for the FDNN dash detector.

Trimmed from C:/Adhitya/Coding/test/FDNN/dash_code/config.py to just what the
inference path (features + dnn + peaks + counter) needs. The geometry/threshold
values are kept verbatim so behaviour matches the FDNN project.
"""

# --- frame rate / timing ---------------------------------------------------
FPS          = 60                 # Marvel Rivals clips are 1920x1080 @ 60 fps
MS_PER_FRAME = 1000.0 / FPS       # 16.667 ms

# --- completion peak decoding (1-D NMS over the probability track) ----------
PEAK_MIN_DIST  = 12    # min frames between two counted peaks (NMS)
PEAK_MATCH_TOL = 8     # a predicted peak matches a GT completion within +/- this
PEAK_THRESHOLD = 0.5   # default NMS confidence cutoff (checkpoint usually overrides)

# --- CNN front-end (spatial features) --------------------------------------
FEAT_DIM      = 512               # resnet18 avgpool features
IMG_SIZE      = 224
FEATURE_BATCH = 64                # frames per forward pass
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- combos (time-window grouping, identical to dash_counter) --------------
# combo window = 450(n-1) + 100 ms -> Double:550ms, Triple:1000ms, Quad:1450ms, Penta:1900ms
COMBO_NAMES = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
