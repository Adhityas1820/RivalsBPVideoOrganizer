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
    "output_mode": pipeline.OUTPUT_MODE,        # 'rename' | 'clip'
    "dash_clip_min": pipeline.DASH_CLIP_MIN,    # 2=Double .. 5=Penta
    "kill_clip_min": pipeline.KILL_CLIP_MIN,    # minimum kills in a streak
    "clip_pad_secs": pipeline.CLIP_PAD_SECS,    # 0.0 .. 10.0
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
        settings = settings or {}
        method = str(settings.get("dash_method", "contour")).lower()
        if method not in ("contour", "fdnn", "hybrid"):
            method = "contour"
        try:
            thr = float(settings.get("fdnn_threshold", pipeline.FDNN_THRESHOLD))
        except (TypeError, ValueError):
            thr = pipeline.FDNN_THRESHOLD
        thr = max(0.05, min(0.95, thr))

        mode = str(settings.get("output_mode", "rename")).lower()
        if mode not in ("rename", "clip"):
            mode = "rename"
        try:
            dash_min = int(settings.get("dash_clip_min", pipeline.DASH_CLIP_MIN))
        except (TypeError, ValueError):
            dash_min = pipeline.DASH_CLIP_MIN
        dash_min = dash_min if dash_min in (2, 3, 4, 5) else pipeline.DASH_CLIP_MIN
        try:
            kill_min = int(settings.get("kill_clip_min", pipeline.KILL_CLIP_MIN))
        except (TypeError, ValueError):
            kill_min = pipeline.KILL_CLIP_MIN
        kill_min = max(1, kill_min)
        try:
            pad = float(settings.get("clip_pad_secs", pipeline.CLIP_PAD_SECS))
        except (TypeError, ValueError):
            pad = pipeline.CLIP_PAD_SECS
        pad = max(0.0, min(10.0, pad))

        self._settings = {
            "dash_method": method, "fdnn_threshold": thr,
            "output_mode": mode, "dash_clip_min": dash_min,
            "kill_clip_min": kill_min, "clip_pad_secs": pad,
        }
        _save_settings(self._settings)
        return self.get_settings()

    # ── file selection ──────────────────────────────────────────────────────

    def pick_files(self):
        ftypes = ("Video files (*.mp4;*.avi;*.mov;*.mkv;*.webm)", "All files (*.*)")
        paths = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=True, file_types=ftypes)
        if not paths:
            return self._files_payload()   # dialog cancelled — keep selection
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

    # ── processing ───────────────────────────────────────────────────────────

    def start(self, options):
        if not self._videos:
            return {"ok": False, "error": "No files selected."}
        options = dict(options or {})
        # Inject the persisted settings unless the UI already supplied them.
        for key in ("dash_method", "fdnn_threshold", "output_mode",
                    "dash_clip_min", "kill_clip_min", "clip_pad_secs"):
            options.setdefault(key, self._settings.get(key))
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
