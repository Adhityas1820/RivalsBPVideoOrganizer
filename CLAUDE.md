# Marvel Rivals Classifier — repo notes

Long-term project memory lives at:
`C:\Adhitya\Obsidian\AI memories\MarvelRivalsClassifier\`
Read its STATUS.md and decisions.md before starting; offer to update them when done.
(The root C:\Adhitya\CLAUDE.md explains the general convention.)

## Project-specific facts

- **Frontend (current):** `python app.py` — pywebview desktop app with an HTML/CSS/JS
  UI in `web/` (index.html, style.css, app.js). The Python↔JS bridge lives in `app.py`;
  all processing is in `pipeline.py`. Package to a double-click `.exe` with
  `pyinstaller MarvelRivalsClassifier.spec --noconfirm` → `dist/MarvelRivalsClassifier/`.
- **pipeline.py** holds the whole processing pipeline (extracted from the old GUI,
  behaviour-identical): `run_pipeline(videos, options, emit)` streams `progress`/`stage`/
  `log`/`done`/`error` events to whatever frontend drives it.
- Legacy entry point: `python main.py` (Tkinter GUI) — kept as reference/fallback.
- Python 3.10+, deps in `requirements.txt` (+ `pywebview`, `pyinstaller` for the app/exe).
- Map detection = fine-tuned **ResNet18** (`train_model.py` → `models/map_classifier.pth`).
- Kills / dashes / combos = **classical OpenCV**, NOT ML. Logic is tied to a
  **1920×1080 @ 60fps** HUD layout — changing resolution or the game's HUD breaks it.
- Combos are time-grouped dash chains (Double/Triple/Quad/Penta) in `dash_counter.py`.
- **Dash detection has 3 selectable methods** (Settings page → gear icon):
  **Contour** (classical OpenCV, default), **FDNN** (neural net), **Hybrid**
  (run both, keep the higher count per clip). FDNN/Hybrid expose a threshold slider.
  - FDNN = vendored from `C:/Adhitya/Coding/test/FDNN`, lives in the `fdnn/` package
    (ResNet18 features → SDSNN temporal head → 1-D NMS peaks). `fdnn.counter.count_dashes_fdnn`
    returns the **same 5-tuple** as `dash_counter.count_dashes`, so they're drop-in.
  - Models: `models/sdsnn.pt` (SDSNN checkpoint) + `models/resnet18_imagenet.pth`
    (bundled ImageNet backbone, so the .exe works offline).
  - Method/threshold are chosen in the UI, persisted to `settings.json`, passed to
    `run_pipeline` via `options['dash_method']` / `options['fdnn_threshold']`. Contour
    runs in the mp pool; FDNN runs in the main process (loads a torch model).
- Output: clips renamed `MapName - Xd - [Combo] - Yk.ext` into `final/`.
- Game-mode pre-filter exists but is off by default (`USE_DOMINATION_FILTER = False`).
- Lots of `test_*` / `assist_*` / `garbage/` scripts are experiments, not the pipeline.
