"""
test_dash_counter2.py
---------------------
At the 2-second mark, finds the largest white contour in the search area
and locks onto its bounding box as "slot x".

Counts a dash when slot x's white ratio drops from >= HIGH_THRESH to < LOW_THRESH.

Output saved to: test/dash_annotated2.mp4

Usage:
    python test_dash_counter2.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
#VIDEO_IN  = "unsorted_videos\\hall of djalia.mp4"
VIDEO_IN  = "unsorted_videos\\arakko 3.mp4"

VIDEO_OUT = "test/dash_annotated2.mp4"

PROCESS_FPS = 60

# Two candidate slots — whichever has more LSHIFT label text is the dash slot
SLOT2_SEARCH = (1575, 1625, 965, 1000)   # usual dash position
SLOT3_SEARCH = (1500, 1550, 965, 1000)   # dash when Jump occupies slot 2
SLOT2_LABEL  = (1575, 1625, 1030, 1050)
SLOT3_LABEL  = (1500, 1550, 1030, 1050)
SLOT_DETECT_FRAMES = 30                  # early frames sampled to pick the slot

RIGHT_SAMPLE_SEC = 7.95   # timestamp to sample right slot (slot2) contour
LEFT_SAMPLE_SEC  = 12.04  # timestamp to sample left slot (slot3) contour

WHITE_THRESH       = 200
WHITE_RATIO_THRESH = 1.0

STABLE_FRAMES      = 1
OFF_FRAMES         = 5
DASH_REARM_SECS    = 0.3
DASH_FLASH_SECS    = 1.0
ZOOM_LOW_THRESH    = 0.2

ZOOM = 10

# Separate contour files for each slot position
RIGHT_CONTOUR_PATH = "reference pictures/slot_x_contour_right.npy"
LEFT_CONTOUR_PATH  = "reference pictures/slot_x_contour_left.npy"

# Combo detection — label is "grey" when white pixel ratio drops below threshold
LABEL_WHITE_THRESH  = 0.1   # label white ratio below this → ability on cooldown
LABEL_STABLE_FRAMES = 10    # consecutive frames label must hold state before it flips
COMBO_GAP_SECS     = 0.9    # max gap between consecutive dashes in the same combo
COMBO_NAMES        = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}
# ---------------------------------------------------------------------------


def label_white_ratio(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (gray > WHITE_THRESH).sum() / max(gray.size, 1)




def white_ratio_in_contours(frame, contours, x0, y0, x1, y1):
    """Measure white ratio only inside the contour shapes, not the whole bounding box."""
    h, w = frame.shape[:2]
    region = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Build a mask of pixels that are inside any contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    shifted_back = [(c - np.array([[[x0, y0]]])).astype(np.int32) for c in contours]
    cv2.drawContours(mask, shifted_back, -1, 255, thickness=cv2.FILLED)

    total_pixels = mask.sum() // 255
    if total_pixels == 0:
        return False, 0.0

    white_pixels = ((gray > WHITE_THRESH) & (mask > 0)).sum()
    ratio = white_pixels / total_pixels
    return ratio >= WHITE_RATIO_THRESH, ratio


def find_all_contours(frame, x0, y0, x1, y1):
    """Find all white contours AND their holes in the search region.
    Returns (list of shifted contours, combined bounding bbox) or ([], None)."""
    h, w = frame.shape[:2]
    region = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    # RETR_TREE captures outer contours AND inner holes
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None
    # Shift all contours to full-frame coords
    shifted = [c + np.array([[[x0, y0]]]) for c in contours]
    # Combined bounding box across all contours
    all_pts = np.concatenate(contours)
    rx, ry, rw, rh = cv2.boundingRect(all_pts)
    bbox = (x0 + rx, x0 + rx + rw, y0 + ry, y0 + ry + rh)
    return shifted, bbox


def fmt_timestamp(seconds):
    m   = int(seconds) // 60
    s   = seconds % 60
    return f"{m}:{s:07.4f}"


def main():
    Path("test").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {VIDEO_IN}")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(src_fps / PROCESS_FPS))
    flash_frames   = int(DASH_FLASH_SECS * src_fps)
    sample_frame   = int(RIGHT_SAMPLE_SEC * src_fps)

    # --- Detect which slot has the LSHIFT label ---
    ratio2_acc = ratio3_acc = 0.0
    sampled = 0
    while sampled < SLOT_DETECT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        ratio2_acc += label_white_ratio(frame, *SLOT2_LABEL)
        ratio3_acc += label_white_ratio(frame, *SLOT3_LABEL)
        sampled += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    search_region = SLOT2_SEARCH if ratio2_acc >= ratio3_acc else SLOT3_SEARCH
    slot_name     = "slot2" if ratio2_acc >= ratio3_acc else "slot3"
    print(f"Dash slot : {slot_name}  (slot2: {ratio2_acc/max(sampled,1):.3f}, slot3: {ratio3_acc/max(sampled,1):.3f})")

    # --- Load or detect contours for BOTH slots ---
    def load_or_detect(path_str, sample_sec, region):
        p = Path(path_str)
        if p.exists():
            data = np.load(str(p), allow_pickle=True)
            contours = list(data)
            print(f"Loaded contour from {p}  ({len(contours)} contour(s))")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_sec * src_fps))
            ret, s = cap.read()
            contours = []
            if ret:
                rx0, rx1, ry0, ry1 = region
                contours, _ = find_all_contours(s, rx0, ry0, rx1, ry1)
            if contours:
                np.save(str(p), np.array(contours, dtype=object))
                print(f"Detected and saved contour to {p}  ({len(contours)} contour(s))")
            else:
                print(f"[WARN] No white contours found for {p.name}")
        return contours

    right_contours = load_or_detect(RIGHT_CONTOUR_PATH, RIGHT_SAMPLE_SEC, SLOT2_SEARCH)
    left_contours  = load_or_detect(LEFT_CONTOUR_PATH,  LEFT_SAMPLE_SEC,  SLOT3_SEARCH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    is_right      = (slot_name == "slot2")
    slot_contours = right_contours if is_right else left_contours

    if slot_contours:
        rx, ry, rw, rh = cv2.boundingRect(np.concatenate(slot_contours).astype(np.int32))
        slot_bbox = (rx, rx + rw, ry, ry + rh)
    else:
        sx0s, sx1s, sy0s, sy1s = search_region
        slot_bbox = (sx0s, sx1s, sy0s, sy1s)

    sx0, sx1, sy0, sy1 = slot_bbox
    print(f"Source  : {VIDEO_IN}")
    print(f"Size    : {fw}x{fh}  @ {src_fps:.1f} fps")
    print(f"Slot x  : x {sx0}-{sx1}  y {sy0}-{sy1}")
    print(f"Output  : {VIDEO_OUT}\n")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = cv2.VideoWriter(
        VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (fw, fh),
    )

    rearm_frames     = int(DASH_REARM_SECS * src_fps)
    save_frame_path  = None
    log_path         = VIDEO_OUT.replace(".mp4", "_hud_log.txt")
    log_file         = open(log_path, "w")
    stable_state     = False
    candidate_state  = False
    candidate_streak = 0
    off_streak       = 0
    was_off          = True
    total_dashes     = 0
    timestamps       = []
    dash_times_sec   = []      # raw float seconds for delta computation
    dash_log         = []      # list of (dash_num, armed_slotX_ratio, trigger_zoom_ratio)
    flash_left       = 0
    cur_ratio        = 0.0
    zoom_ratio       = 0.0
    white_state      = False
    rearm_at         = 0
    read_idx         = 0

    # Combo state
    active_label_region    = SLOT2_LABEL if is_right else SLOT3_LABEL
    label_ratio            = 0.0
    label_is_grey          = False   # raw per-frame
    label_candidate_grey   = False
    label_candidate_streak = 0
    label_stable_grey      = False   # debounced, used for combo logic
    prev_label_stable_grey = False
    combo_count            = 0
    last_combo_dash_sec    = None
    combos                 = []   # list of (count, name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx % frame_interval == 0:
            h_f, w_f = frame.shape[:2]
            white_state, cur_ratio = white_ratio_in_contours(frame, slot_contours, sx0, sy0, sx1, sy1)

            # Track label grey/white state for combo detection
            label_ratio   = label_white_ratio(frame, *active_label_region)
            label_is_grey = label_ratio < LABEL_WHITE_THRESH

            # Stable label state (debounced)
            if label_is_grey == label_candidate_grey:
                label_candidate_streak += 1
            else:
                label_candidate_grey   = label_is_grey
                label_candidate_streak = 1
            prev_label_stable_grey = label_stable_grey
            if label_candidate_streak >= LABEL_STABLE_FRAMES:
                label_stable_grey = label_candidate_grey

            # Label just went white → finalize any open combo
            if prev_label_stable_grey and not label_stable_grey:
                if combo_count >= 2:
                    name = COMBO_NAMES.get(combo_count, f"{combo_count}x")
                    combos.append((combo_count, name))
                combo_count         = 0
                last_combo_dash_sec = None

            # Zoom ratio — white ratio of the bounding box EXCLUDING slot x contour pixels
            h_f2, w_f2 = frame.shape[:2]
            zoom_crop  = frame[min(sy0,h_f2):min(sy1,h_f2), min(sx0,w_f2):min(sx1,w_f2)]
            if zoom_crop.size > 0:
                gray_z = cv2.cvtColor(zoom_crop, cv2.COLOR_BGR2GRAY)
                # Build exclusion mask for slot x contour pixels
                excl_mask = np.zeros(gray_z.shape, dtype=np.uint8)
                if slot_contours:
                    shifted = [(c - np.array([[[sx0, sy0]]])).astype(np.int32) for c in slot_contours]
                    cv2.drawContours(excl_mask, shifted, -1, 255, thickness=cv2.FILLED)
                outside = excl_mask == 0
                total_outside = outside.sum()
                if total_outside > 0:
                    zoom_ratio = ((gray_z > WHITE_THRESH) & outside).sum() / total_outside
                else:
                    zoom_ratio = 0.0
            else:
                zoom_ratio = 0.0

            # Count dash: slot x IS white AND zoom box is below threshold RIGHT NOW
            zoom_low = zoom_ratio < ZOOM_LOW_THRESH
            if white_state and zoom_low and read_idx >= rearm_at and was_off:
                total_dashes += 1
                t_sec = read_idx / src_fps
                ts = fmt_timestamp(t_sec)
                timestamps.append(ts)
                dash_times_sec.append(t_sec)
                dash_log.append((total_dashes, cur_ratio, zoom_ratio, ts))
                save_frame_path = f"test/dash_{total_dashes}_frame.jpg"
                flash_left = flash_frames
                rearm_at   = read_idx + rearm_frames
                was_off    = False

                # Accumulate into combo while label is grey
                if label_stable_grey:
                    if last_combo_dash_sec is None or (t_sec - last_combo_dash_sec) <= COMBO_GAP_SECS:
                        combo_count += 1
                    else:
                        # Gap too large — finalize previous combo, start new one
                        if combo_count >= 2:
                            name = COMBO_NAMES.get(combo_count, f"{combo_count}x")
                            combos.append((combo_count, name))
                        combo_count = 1
                    last_combo_dash_sec = t_sec

            if not white_state:
                off_streak += 1
                if off_streak >= OFF_FRAMES:
                    was_off = True
            else:
                off_streak = 0

            if white_state == candidate_state:
                candidate_streak += 1
            else:
                candidate_state  = white_state
                candidate_streak = 1

            if candidate_streak == STABLE_FRAMES:
                stable_state = candidate_state

        # --- Draw annotations ---

        # Search area (dim grey)
        sr_x0, sr_x1, sr_y0, sr_y1 = search_region
        cv2.rectangle(frame, (sr_x0, sr_y0), (sr_x1, sr_y1), (60, 60, 60), 1)

        # Inactive slot — dim rectangle
        ix0, ix1, iy0, iy1 = SLOT3_SEARCH if is_right else SLOT2_SEARCH
        cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), (60, 60, 40), 1)

        # Active slot x — white when triggered, grey otherwise
        box_color = (255, 255, 255) if white_state else (80, 80, 80)
        if slot_contours:
            cv2.drawContours(frame, [c.astype(np.int32) for c in slot_contours], -1, box_color, 1)
        cv2.rectangle(frame, (sx0, sy0), (sx1, sy1), box_color, 1)
        cv2.putText(frame, f"slot x ({slot_name})", (sx0, sy0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # Zoomed preview — top right (slot x)
        crop_preview = frame[min(sy0,fh):min(sy1,fh), min(sx0,fw):min(sx1,fw)]
        if crop_preview.size > 0:
            zh = max(crop_preview.shape[0] * ZOOM, 1)
            zw = max(crop_preview.shape[1] * ZOOM, 1)
            zoomed = cv2.resize(crop_preview, (zw, zh), interpolation=cv2.INTER_NEAREST)
            px, py = fw - zw - 20, 60
            if py+zh <= fh and px >= 0:
                frame[py:py+zh, px:px+zw] = zoomed
                cv2.rectangle(frame, (px, py), (px+zw, py+zh), box_color, 2)
                cv2.putText(frame, "slot x (zoomed)", (px, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Zoomed preview — label (right under slot x)
            lx0, lx1, ly0, ly1 = active_label_region
            label_preview = frame[min(ly0,fh):min(ly1,fh), min(lx0,fw):min(lx1,fw)]
            if label_preview.size > 0:
                lzh = max(label_preview.shape[0] * ZOOM, 1)
                lzw = max(label_preview.shape[1] * ZOOM, 1)
                lzoomed = cv2.resize(label_preview, (lzw, lzh), interpolation=cv2.INTER_NEAREST)
                lpx, lpy = fw - lzw - 20, py + zh + 20
                label_color = (100, 100, 100) if label_stable_grey else (200, 200, 200)
                if lpy + lzh <= fh and lpx >= 0:
                    frame[lpy:lpy+lzh, lpx:lpx+lzw] = lzoomed
                    cv2.rectangle(frame, (lpx, lpy), (lpx+lzw, lpy+lzh), label_color, 2)
                    cv2.putText(frame, f"label ({label_ratio:.3f})", (lpx, lpy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)

        # HUD text
        t_now = read_idx / src_fps
        hud_line = (
            f"{fmt_timestamp(t_now)}  slotX: {cur_ratio:.3f} {'W' if white_state else '-'}"
            f"  zoom: {zoom_ratio:.3f} {'LOW' if zoom_ratio < ZOOM_LOW_THRESH else '-'}"
            f"  {'rearm' if read_idx < rearm_at else 'ready'}"
            f"  was_off: {'Y' if was_off else 'N'}"
            f"  dashes: {total_dashes}"
            f"  lbl: {label_ratio:.3f} {'GREY' if label_stable_grey else ('grey?' if label_is_grey else 'white')}"
            f"  combo: {combo_count}"
        )
        cv2.putText(frame, hud_line, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        log_file.write(hud_line + "\n")

        for i, (num, sr, zr, ts) in enumerate(dash_log):
            cv2.putText(frame,
                f"dash {num}  [{ts}]  slotX:{sr:.3f}  zoom:{zr:.3f}",
                (20, 70 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        if flash_left > 0:
            cv2.putText(frame, "DASH!",
                        (fw // 2 - 100, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6, cv2.LINE_AA)
            flash_left -= 1

        if save_frame_path:
            cv2.imwrite(save_frame_path, frame)
            save_frame_path = None

        out.write(frame)

        read_idx += 1
        if read_idx % int(src_fps * 10) == 0:
            pct = 100 * read_idx // total_frames
            print(f"\r  {pct:3d}%  [{fmt_timestamp(read_idx / src_fps)}]", end="", flush=True)

    # Finalize any open combo at end of video
    if combo_count >= 2:
        name = COMBO_NAMES.get(combo_count, f"{combo_count}x")
        combos.append((combo_count, name))

    cap.release()
    log_file.close()
    print(f"HUD log     : {log_path}")
    out.release()

    print(f"\n\nDone.")
    print(f"Total dashes : {total_dashes}")
    if dash_times_sec:
        print("Dash timestamps:")
        for i, (t, ts) in enumerate(zip(dash_times_sec, timestamps)):
            if i == 0:
                print(f"  dash 1: {ts}")
            else:
                delta = t - dash_times_sec[i - 1]
                print(f"  dash {i + 1}: {timestamps[i - 1]}, [{delta:.3f}s] {ts}")
    else:
        print("Timestamps   : none")

    print(f"\nCombos detected: {len(combos)}")
    for i, (count, name) in enumerate(combos, 1):
        print(f"  combo {i}: {name} ({count} dashes)")

    print(f"Saved to     : {VIDEO_OUT}")


if __name__ == "__main__":
    main()
