"""
debug_crop.py
-------------
Two modes for tuning HUD region coordinates.

AUTO mode:
  Opens a window with the frame at SEEK_SEC. Click and drag to draw a
  selection box. Release to print the pixel coordinates. Press 'r' to
  reset, 'q' or Escape to quit.

MANUAL mode:
  Draws the three named boxes (SLOT, LABEL, FILL) using the hardcoded
  coordinates below and saves the result to debug_crops/.

Set MODE = "auto" or "manual" below.
"""

import cv2
import numpy as np
import tkinter as tk
from pathlib import Path


def copy_to_clipboard(text: str):
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODE     = "auto"   # "auto" or "manual"
VIDEO_IN = r"unsorted_videos\celestial husk 1 - 5 kills - 9 dashes.mp4"
SEEK_SEC = 7.5
OUT_DIR  = "debug_crops"

# ── Manual mode boxes ────────────────────────────────────────────────────────
SLOT_X0,  SLOT_X1,  SLOT_Y0,  SLOT_Y1  = 25,   600,  900, 1060
LABEL_X0, LABEL_X1, LABEL_Y0, LABEL_Y1 = 1180, 1210,  405,  440
FILL_X0,  FILL_X1,  FILL_Y0,  FILL_Y1  = 750,  1175,  970, 1050
# ---------------------------------------------------------------------------


def load_frame(video_in: str, seek_sec: float):
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_sec * fps))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def run_auto(frame):
    """Interactive selection: click-drag to measure a region."""
    clone     = frame.copy()
    display   = frame.copy()
    drawing   = False
    start_pt  = None
    end_pt    = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, start_pt, end_pt, display

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing  = True
            start_pt = (x, y)
            end_pt   = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_pt  = (x, y)
            display = clone.copy()
            cv2.rectangle(display, start_pt, end_pt, (0, 255, 0), 1)
            # Show live coords
            x0, y0 = min(start_pt[0], end_pt[0]), min(start_pt[1], end_pt[1])
            x1, y1 = max(start_pt[0], end_pt[0]), max(start_pt[1], end_pt[1])
            cv2.putText(display, f"x0={x0} y0={y0}  x1={x1} y1={y1}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing  = False
            end_pt   = (x, y)
            display  = clone.copy()
            x0, y0 = min(start_pt[0], end_pt[0]), min(start_pt[1], end_pt[1])
            x1, y1 = max(start_pt[0], end_pt[0]), max(start_pt[1], end_pt[1])
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
            label = f"x0={x0} x1={x1} y0={y0} y1={y1}  ({x1-x0}x{y1-y0}px)"
            cv2.putText(display, label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            coords = f"{x0}, {x1}, {y0}, {y1}"
            copy_to_clipboard(coords)
            print(f"\nSelected: x0={x0}, x1={x1}, y0={y0}, y1={y1}  "
                  f"(width={x1-x0}, height={y1-y0})")
            print(f"  → copied to clipboard: {coords}")

    h, w = frame.shape[:2]
    win = "AUTO — drag to select  |  r=reset  q/Esc=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1600), min(h, 900))
    cv2.setMouseCallback(win, mouse_cb)

    print("Auto mode: drag a box on the frame to get pixel coordinates.")
    print("Press 'r' to reset, 'q' or Escape to quit.\n")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            display  = clone.copy()
            start_pt = None
            end_pt   = None
            print("Reset.")
        elif key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


def run_manual(frame, video_name: str):
    annotated = frame.copy()

    cv2.rectangle(annotated, (SLOT_X0, SLOT_Y0), (SLOT_X1, SLOT_Y1), (255, 255, 255), 2)
    cv2.putText(annotated, "SLOT", (SLOT_X0, SLOT_Y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(annotated, (LABEL_X0, LABEL_Y0), (LABEL_X1, LABEL_Y1), (0, 255, 0), 2)
    cv2.putText(annotated, "LABEL", (LABEL_X0, LABEL_Y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.rectangle(annotated, (FILL_X0, FILL_Y0), (FILL_X1, FILL_Y1), (0, 220, 220), 2)
    cv2.putText(annotated, "FILL", (FILL_X0, FILL_Y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    Path(OUT_DIR).mkdir(exist_ok=True)
    out_path = Path(OUT_DIR) / f"{Path(video_name).stem}_boxes.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved: {out_path}")


def main():
    frame = load_frame(VIDEO_IN, SEEK_SEC)
    if frame is None:
        print(f"Could not read frame at {SEEK_SEC}s from {VIDEO_IN}")
        return

    h, w = frame.shape[:2]
    print(f"{Path(VIDEO_IN).name}  ({w}x{h})  @ {SEEK_SEC}s")

    if MODE == "auto":
        run_auto(frame)
    else:
        run_manual(frame, VIDEO_IN)
        print("Done. Open debug_crops/ to review.")


if __name__ == "__main__":
    main()
