# Marvel Rivals Classifier

Automatically organizes Marvel Rivals gameplay clips by map, and counts kills and dashes for each clip.

## What it does

1. You pick your raw gameplay clips in the app
2. It counts kills and dashes in each clip
3. It detects which map the clip is from using a trained ResNet18 model
4. It renames and saves each clip as `MapName - Xd - Yk.mp4` (or with combo labels e.g. `MapName - 3d - Double - 5k.mp4`) into a `final/` folder
5. You can filter, sort, and download a ZIP of the results

## Requirements

**Python 3.10+**

Install dependencies:

```
pip install -r requirements.txt
```

## Files needed

```
main.py
kill_counter.py
dash_counter.py
game_mode_select.py
requirements.txt
models/
  map_classifier.pth
  scoreboard_slot1_contour.npy
  scoreboard_slot2_contour.npy
  slot_x_contour_right.npy
  slot_x_contour_left.npy
```

## How to run

```
python main.py
```

1. Click **Browse** and select your video clips (MP4, AVI, MOV, MKV, or WEBM)
2. Click **Process Clips**
3. Wait for the pipeline to finish — this takes a few minutes depending on clip count and length
4. Use the filter and sort options to browse results
5. Click **Save** on individual clips or **Download ZIP** to grab them all

## Output

Clips are saved to a `final/` folder in the same directory as `main.py`, renamed with this format:

```
MapName - Xd - Yk.ext
MapName - Xd - ComboName - Yk.ext
```

For example:
```
Royal Palace - 4d - Double - 7k.mp4
Krakoa - 2d - 3k.mp4
```

## Notes

- Videos must be 1920×1080 at 60fps for accurate kill and dash detection
- The `final/` folder accumulates clips across sessions — the ZIP only includes clips from the current run
- A GPU will speed up map classification but is not required
