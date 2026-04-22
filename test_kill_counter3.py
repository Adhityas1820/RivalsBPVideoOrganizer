"""
test_kill_counter3.py
---------------------
Kill feed row detection (test_kill_counter2 logic) with scoreboard awareness.

When the scoreboard is open:
  - No kills are counted.
  - The stable baseline is kept in sync with the current lit count so that
    when the board closes, no false kills are triggered from rows that were
    already lit before the board opened.

Run assist_test_kill_counter.py first to save scoreboard slot contours.

Output: test/test_kill_counter3.mp4

Usage:
    python test_kill_counter3.py
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIDEO_IN  = r"unsorted_videos\hells heaven 2 2k.mp4"
VIDEO_OUT = "test/test_kill_counter3.mp4"

PROCESS_FPS = 60

# Kill feed rows
X_START    = 1625
X_END      = 1875
ROW_H      = 30
ROW_GAP    = 10
ROW_COUNT  = 6
ROW_Y_BASE = 30

ROW_Y_STARTS = [ROW_Y_BASE + i * (ROW_H + ROW_GAP) for i in range(ROW_COUNT)]

WHITE_THRESH    = 200
ROW_WHITE_RATIO = 0.20
DIST_MIN        = 2
DIST_MAX        = 10

STABLE_FRAMES    = 10
KILL_TIMER_SECS  = 5.15  # each kill feed entry lasts ~5s; fire slightly after to confirm gone
KILL_FLASH_SECS  = 1.0
BOARD_LOCKOUT    = 0.2   # seconds after board closes before kills count again

# Ability indicator — triggers a wider DIST_MAX window on yellow→blue transition
ABILITY_X0, ABILITY_X1, ABILITY_Y0, ABILITY_Y1 = 1865, 1870, 943, 949

YELLOW_R_MIN, YELLOW_G_MIN, YELLOW_B_MAX = 150, 150, 100
BLUE_B_MIN = 200   # #abc4fe has B=254 — detect blue as: B dominant and B high
ABILITY_COLOR_THRESH = 0.90

DIST_MAX_WIDE      = 90
WIDE_DELAY_SECS    = 1
WIDE_DURATION_SECS = 0.7

# Scoreboard slots
SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1 = 760,  938,  232, 278
SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1 = 1423, 1597, 236, 283

SLOT1_CONTOUR_PATH = "models/scoreboard_slot1_contour.npy"
SLOT2_CONTOUR_PATH = "models/scoreboard_slot2_contour.npy"

SLOT_WHITE_RATIO = 0.80   # fraction of contour pixels that must be white
BOX_MAX_RATIO    = 0.30   # outer box ratio must stay below this

BOX_COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 128, 0),
]
# ---------------------------------------------------------------------------



def row_is_lit(crop: np.ndarray, dist_max: int = DIST_MAX) -> tuple:
    """Returns (is_lit, ratio, dist, n_cnts, cnt_info) where cnt_info = [(area, contour), ...]."""
    b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    white_mask = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
    ratio = white_mask.sum() / white_mask.size
    mask = white_mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_info = sorted([(int(cv2.contourArea(c)), c) for c in cnts], key=lambda x: x[0], reverse=True)
    n_cnts = len(cnt_info)
    if ratio < ROW_WHITE_RATIO:
        return False, ratio, -1.0, n_cnts, cnt_info
    top2 = cnt_info[:2]
    if len(top2) < 2:
        return False, ratio, -1.0, n_cnts, cnt_info
    pts1 = top2[0][1].reshape(-1, 2).astype(np.float32)
    pts2 = top2[1][1].reshape(-1, 2).astype(np.float32)
    dists = np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2)
    dist = float(dists.min())
    return DIST_MIN <= dist <= dist_max, ratio, dist, n_cnts, cnt_info


def load_contours(path):
    p = Path(path)
    if not p.exists():
        return []
    return list(np.load(str(p), allow_pickle=True))


def slot_is_white(frame, contours, x0, x1, y0, y1) -> tuple:
    """Returns (is_on, contour_ratio, box_ratio) — contour-based detection."""
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return False, 0.0, 0.0
    b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    white = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
    box_ratio = white.sum() / white.size
    if contours:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        shifted = [(c - np.array([[[x0, y0]]])).astype(np.int32) for c in contours]
        cv2.drawContours(mask, shifted, -1, 255, thickness=cv2.FILLED)
        total = mask.sum() // 255
        if total == 0:
            return False, 0.0, box_ratio
        contour_ratio = ((white) & (mask > 0)).sum() / total
    else:
        contour_ratio = box_ratio
    is_on = contour_ratio >= SLOT_WHITE_RATIO and box_ratio <= BOX_MAX_RATIO
    return is_on, contour_ratio, box_ratio


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def main():
    Path("test").mkdir(exist_ok=True)

    contours1 = load_contours(SLOT1_CONTOUR_PATH)
    contours2 = load_contours(SLOT2_CONTOUR_PATH)
    print(f"Scoreboard slot 1 : {len(contours1)} contour(s) from {SLOT1_CONTOUR_PATH}")
    print(f"Scoreboard slot 2 : {len(contours2)} contour(s) from {SLOT2_CONTOUR_PATH}")

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {VIDEO_IN}")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(src_fps / PROCESS_FPS))
    flash_frames   = int(KILL_FLASH_SECS * src_fps)

    print(f"Source  : {VIDEO_IN}")
    print(f"Size    : {w}x{h}  @ {src_fps:.1f} fps")
    print(f"Output  : {VIDEO_OUT}\n")

    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (w, h),
    )

    n                = len(ROW_Y_STARTS)
    cur_has          = [False] * n
    cur_ratios       = [0.0]   * n
    cur_dists        = [-1.0]  * n
    cur_cnts         = [[]]    * n
    candidate_count     = 0
    streak              = 0
    pending_kill_timers = []   # list of expire_at timestamps, one per pending kill

    board_on       = False
    prev_board_on  = False
    s1_on          = False
    s2_on          = False
    s1_cratio      = 0.0
    s2_cratio      = 0.0
    kill_locked_until = 0.0

    kills      = 0
    flash_left = 0
    read_idx   = 0

    # Ability-indicator state
    ability_is_yellow  = False
    wide_dist_start_at = -1.0
    wide_dist_end_at   = -1.0
    cur_dist_max       = DIST_MAX
    ability_yellow_r   = 0.0
    ability_blue_r     = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx % frame_interval == 0:
            hf, wf = frame.shape[:2]
            t_sec  = read_idx / src_fps

            # --- Ability indicator: detect yellow→blue transition ---
            ab_crop = frame[min(ABILITY_Y0,hf):min(ABILITY_Y1,hf),
                            min(ABILITY_X0,wf):min(ABILITY_X1,wf)]
            if ab_crop.size > 0:
                b_ab, g_ab, r_ab = ab_crop[:,:,0], ab_crop[:,:,1], ab_crop[:,:,2]
                n_px = ab_crop.shape[0] * ab_crop.shape[1]
                ability_yellow_r = ((r_ab > YELLOW_R_MIN) & (g_ab > YELLOW_G_MIN) & (b_ab < YELLOW_B_MAX)).sum() / n_px
                ability_blue_r   = ((b_ab > BLUE_B_MIN) & (b_ab > r_ab) & (b_ab > g_ab)).sum() / n_px
                is_yellow_now = ability_yellow_r >= ABILITY_COLOR_THRESH
                is_blue_now   = ability_blue_r   >= ABILITY_COLOR_THRESH
                if ability_is_yellow and is_blue_now:
                    wide_dist_start_at = t_sec + WIDE_DELAY_SECS
                    wide_dist_end_at   = t_sec + WIDE_DELAY_SECS + WIDE_DURATION_SECS
                ability_is_yellow = is_yellow_now
            cur_dist_max = DIST_MAX_WIDE if (wide_dist_start_at <= t_sec <= wide_dist_end_at) else DIST_MAX

            # --- Scoreboard detection ---
            s1_on, s1_cratio, _ = slot_is_white(frame, contours1, SLOT1_X0, SLOT1_X1, SLOT1_Y0, SLOT1_Y1)
            s2_on, s2_cratio, _ = slot_is_white(frame, contours2, SLOT2_X0, SLOT2_X1, SLOT2_Y0, SLOT2_Y1)
            prev_board_on    = board_on
            board_on         = s1_on or s2_on

            # Board just closed — start lockout timer
            if prev_board_on and not board_on:
                kill_locked_until = t_sec + BOARD_LOCKOUT

            # --- Kill feed rows ---
            lit = 0
            for i, y0 in enumerate(ROW_Y_STARTS):
                y1   = min(y0 + ROW_H, hf)
                crop = frame[y0:y1, min(X_START,wf):min(X_END,wf)]
                if crop.size == 0:
                    continue
                lit_row, ratio, dist, n_cnts, cnt_info = row_is_lit(crop, cur_dist_max)
                cur_has[i]    = lit_row
                cur_ratios[i] = ratio
                cur_dists[i]  = dist
                cur_cnts[i]   = cnt_info
                if lit_row:
                    lit += 1

            if lit == candidate_count:
                streak += 1
            else:
                candidate_count = lit
                streak          = 1

            locked = board_on or t_sec < kill_locked_until

            # Count kills immediately when lit count stabilises with new uncovered rows
            if streak == STABLE_FRAMES and not locked:
                uncovered = candidate_count - len(pending_kill_timers)
                if uncovered > 0:
                    kills      += uncovered
                    flash_left  = flash_frames
                    for _ in range(uncovered):
                        pending_kill_timers.append(t_sec + KILL_TIMER_SECS)
                        print(f"  KILL  @ {fmt_timestamp(t_sec)}  (total: {kills})")

            # Fire expired timers — only to detect replacement kills (simultaneous fade/replace)
            if not locked:
                fired = sum(1 for e in pending_kill_timers if t_sec >= e)
                if fired > 0:
                    pending_kill_timers[:] = [e for e in pending_kill_timers if t_sec < e]
                    uncovered = candidate_count - len(pending_kill_timers)
                    if uncovered > 0:
                        kills      += uncovered
                        flash_left  = flash_frames
                        for _ in range(uncovered):
                            pending_kill_timers.append(t_sec + KILL_TIMER_SECS)
                            print(f"  KILL  @ {fmt_timestamp(t_sec)}  [replacement]  (total: {kills})")

        # --- Draw kill feed boxes + contour closest-point lines ---
        for i, y0 in enumerate(ROW_Y_STARTS):
            y1    = y0 + ROW_H
            color = BOX_COLORS[i]
            thick = 3 if cur_has[i] else 1
            cv2.rectangle(frame, (X_START, y0), (X_END, y1), color, thick)
            top2 = cur_cnts[i][:2]
            if len(top2) == 2:
                pts1 = top2[0][1].reshape(-1, 2).astype(np.float32)
                pts2 = top2[1][1].reshape(-1, 2).astype(np.float32)
                dists = np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2)
                idx = np.unravel_index(dists.argmin(), dists.shape)
                line_color = (0, 255, 0) if DIST_MIN <= cur_dists[i] <= cur_dist_max else (0, 0, 255)
                p1 = (int(X_START + pts1[idx[0], 0]), int(y0 + pts1[idx[0], 1]))
                p2 = (int(X_START + pts2[idx[1], 0]), int(y0 + pts2[idx[1], 1]))
                cv2.line(frame, p1, p2, line_color, 1)
                cv2.circle(frame, p1, 3, line_color, -1)
                cv2.circle(frame, p2, 3, line_color, -1)

        # --- Draw scoreboard slots ---
        col1 = (0, 255, 0) if s1_on else (0, 0, 255)
        col2 = (0, 255, 0) if s2_on else (0, 0, 255)
        cv2.rectangle(frame, (SLOT1_X0, SLOT1_Y0), (SLOT1_X1, SLOT1_Y1), col1, 2)
        cv2.putText(frame, f"s1:{s1_cratio:.3f}", (SLOT1_X0, SLOT1_Y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col1, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (SLOT2_X0, SLOT2_Y0), (SLOT2_X1, SLOT2_Y1), col2, 2)
        cv2.putText(frame, f"s2:{s2_cratio:.3f}", (SLOT2_X0, SLOT2_Y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col2, 2, cv2.LINE_AA)

        # --- HUD line 1: timestamp + per-row ON/ratio ---
        t_now = fmt_timestamp(read_idx / src_fps)
        row_str = "  ".join(f"r{i+1}:{'ON' if cur_has[i] else '--'}({cur_ratios[i]:.2f})" for i in range(n))
        cv2.putText(frame, f"{t_now}  {row_str}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # --- HUD line 2: per-row contour count + top-2 areas + distance ---
        parts = []
        for i in range(n):
            top2 = cur_cnts[i][:2]
            areas = ",".join(str(a) for a, _ in top2)
            d_str = f" d={int(cur_dists[i])}" if cur_dists[i] >= 0 else ""
            parts.append(f"r{i+1}:{len(cur_cnts[i])}c[{areas}{d_str}]")
        cv2.putText(frame, "  ".join(parts),
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 0), 1, cv2.LINE_AA)

        cur_sec     = read_idx / src_fps
        locked_now  = board_on or cur_sec < kill_locked_until
        if board_on:
            board_label = "SCOREBOARD ON"
            board_col   = (0, 255, 0)
        elif cur_sec < kill_locked_until:
            board_label = f"LOCKOUT {kill_locked_until - cur_sec:.2f}s"
            board_col   = (0, 165, 255)
        else:
            board_label = "BOARD OFF"
            board_col   = (0, 0, 255)
        t_now_hud  = read_idx / src_fps
        timers_str = f"  [{','.join(f'{e-t_now_hud:.1f}s' for e in sorted(pending_kill_timers))}]" if pending_kill_timers else ""
        cv2.putText(frame, f"kills:{kills}  lit:{candidate_count}  streak:{streak}/{STABLE_FRAMES}  timers:{len(pending_kill_timers)}{timers_str}",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, board_label,
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, board_col, 2, cv2.LINE_AA)

        if ability_yellow_r >= ABILITY_COLOR_THRESH:
            ab_label, ab_color = "YELLOW", (0, 255, 255)
        elif ability_blue_r >= ABILITY_COLOR_THRESH:
            ab_label, ab_color = "BLUE", (255, 100, 0)
        else:
            ab_label, ab_color = "OTHER", (150, 150, 150)
        cv2.putText(frame,
            f"ability y:{ability_yellow_r:.2f} b:{ability_blue_r:.2f} [{ab_label}]  dmax:{cur_dist_max}",
            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ab_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (ABILITY_X0, ABILITY_Y0), (ABILITY_X1, ABILITY_Y1), ab_color, 1)

        if flash_left > 0:
            cv2.putText(frame, "KILL!",
                        (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6, cv2.LINE_AA)
            flash_left -= 1

        out.write(frame)
        read_idx += 1

        if read_idx % int(src_fps * 10) == 0:
            pct = 100 * read_idx // max(total_frames, 1)
            print(f"\r  {pct:3d}%  [{fmt_timestamp(read_idx/src_fps)}]", end="", flush=True)

    cap.release()
    out.release()
    print(f"\n\nDone.")
    print(f"Total kills : {kills}")
    print(f"Saved to    : {VIDEO_OUT}")


if __name__ == "__main__":
    main()
