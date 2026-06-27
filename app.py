# -*- coding: utf-8 -*-
"""app.py — Marvel Rivals Classifier desktop app (pywebview frontend).

A native desktop window rendering an HTML/CSS/JS UI. All processing is done by
``pipeline.run_pipeline`` (unchanged behaviour); this module is just the bridge
between the web UI and the Python pipeline.

Run with:  python app.py
Build .exe: see build_exe.py / MarvelRivals.spec
"""

import os
import sys
import json
import threading
import multiprocessing as mp
from pathlib import Path

import webview

import pipeline


def _resource_dir() -> Path:
    """Folder that holds bundled resources (web/, models/). Works both when run
    from source and when frozen by PyInstaller (sys._MEIPASS)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _setup_frozen_paths():
    """When packaged as an .exe, point bundled-resource reads at the unpack dir
    and send organised clips to a 'final' folder beside the executable so users
    can find them easily."""
    if not getattr(sys, "frozen", False):
        return
    res = _resource_dir()
    os.chdir(str(res))                       # so 'models/...' relative reads resolve
    out = Path(sys.executable).parent / "final"
    pipeline.SORTED_DIR = str(out)


RES_DIR  = _resource_dir()
WEB_DIR  = RES_DIR / "web"
INDEX    = WEB_DIR / "index.html"


def _settings_path() -> Path:
    """Where to persist user settings. Beside the .exe when frozen (writable),
    else in the project folder."""
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parent
    return base / "settings.json"


DEFAULT_SETTINGS = {
    "dash_method": pipeline.DASH_METHOD,        # 'contour' | 'fdnn' | 'hybrid'
    "fdnn_threshold": pipeline.FDNN_THRESHOLD,  # 0.05 .. 0.95
}


def _load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    try:
        p = _settings_path()
        if p.exists():
            s.update(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass
    return s


def _save_settings(s: dict):
    try:
        _settings_path().write_text(json.dumps(s, indent=2), encoding="utf-8")
    except Exception:
        pass


class Api:
    """Methods on this object are callable from JS as ``pywebview.api.<name>``."""

    def __init__(self):
        self._window = None
        self._videos = []          # list[Path] currently selected
        self._results = []         # last run's result dicts
        self._settings = _load_settings()

    def set_window(self, window):
        self._window = window

    # ── status ─────────────────────────────────────────────────────────────

    def get_status(self):
        return {
            "model_ready": pipeline.model_ready(),
            "fdnn_available": pipeline.fdnn_model_ready(),
            "stages": pipeline.STAGE_META,
        }

    # ── settings ────────────────────────────────────────────────────────────

    def get_settings(self):
        s = dict(self._settings)
        s["fdnn_available"] = pipeline.fdnn_model_ready()
        return s

    def set_settings(self, settings):
        method = str((settings or {}).get("dash_method", "contour")).lower()
        if method not in ("contour", "fdnn", "hybrid"):
            method = "contour"
        try:
            thr = float((settings or {}).get("fdnn_threshold", pipeline.FDNN_THRESHOLD))
        except (TypeError, ValueError):
            thr = pipeline.FDNN_THRESHOLD
        thr = max(0.05, min(0.95, thr))
        self._settings = {"dash_method": method, "fdnn_threshold": thr}
        _save_settings(self._settings)
        return self.get_settings()

    # ── file selection ──────────────────────────────────────────────────────

    def pick_files(self):
        ftypes = ("Video files (*.mp4;*.avi;*.mov;*.mkv;*.webm)", "All files (*.*)")
        paths = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=True, file_types=ftypes)
        if not paths:
            return [c.export() for c in self._clips_payload()]  # unchanged
        self._videos = [Path(p) for p in paths]
        return self._files_payload()

    def add_files(self, paths):
        """Accept paths from a drag-and-drop onto the window."""
        new = [Path(p) for p in paths if Path(p).suffix.lower() in pipeline.VIDEO_EXTENSIONS]
        existing = {str(v) for v in self._videos}
        for p in new:
            if str(p) not in existing:
                self._videos.append(p)
        return self._files_payload()

    def clear_files(self):
        self._videos = []
        return self._files_payload()

    def remove_file(self, name):
        self._videos = [v for v in self._videos if v.name != name]
        return self._files_payload()

    def _files_payload(self):
        return {"files": [{"name": v.name, "path": str(v)} for v in self._videos]}

    def _clips_payload(self):
        return []

    # ── processing ───────────────────────────────────────────────────────────

    def start(self, options):
        if not self._videos:
            return {"ok": False, "error": "No files selected."}
        options = dict(options or {})
        # Inject the persisted dash settings unless the UI already supplied them.
        options.setdefault("dash_method", self._settings.get("dash_method"))
        options.setdefault("fdnn_threshold", self._settings.get("fdnn_threshold"))
        videos = list(self._videos)
        threading.Thread(
            target=self._run, args=(videos, options), daemon=True).start()
        return {"ok": True}

    def _run(self, videos, options):
        def emit(event_type, **fields):
            fields["type"] = event_type
            if event_type == "done":
                self._results = fields.get("results", [])
            payload = json.dumps(fields)
            try:
                self._window.evaluate_js(f"window.__event({payload})")
            except Exception:
                pass
        pipeline.run_pipeline(videos, options, emit)

    # ── export ────────────────────────────────────────────────────────────────

    def save_clip(self, filename):
        dst = self._window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=filename)
        if not dst:
            return {"ok": False}
        dst = dst if isinstance(dst, str) else dst[0]
        ok = pipeline.export_clip(filename, dst)
        return {"ok": ok}

    def save_zip(self, filenames):
        dst = self._window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename="rivals_clips.zip",
            file_types=("ZIP archive (*.zip)",))
        if not dst:
            return {"ok": False}
        dst = dst if isinstance(dst, str) else dst[0]
        n = pipeline.export_zip(filenames, dst)
        return {"ok": n > 0, "count": n}

    def reveal_output(self):
        out = Path(pipeline.SORTED_DIR).resolve()
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out))   # Windows
        except Exception:
            pass
        return {"ok": True}

    def close(self):
        if self._window:
            self._window.destroy()


def main():
    api = Api()
    window = webview.create_window(
        "Marvel Rivals Classifier",
        url=str(INDEX),
        js_api=api,
        width=1040,
        height=760,
        min_size=(820, 600),
        background_color="#0b0a14",
        text_select=False,
    )
    api.set_window(window)

    # Enable file drag-and-drop onto the window where supported.
    def _on_loaded():
        try:
            window.events.loaded -= _on_loaded
        except Exception:
            pass

    webview.start(_on_loaded, gui=None, debug=False)


if __name__ == "__main__":
    mp.freeze_support()
    _setup_frozen_paths()
    main()
