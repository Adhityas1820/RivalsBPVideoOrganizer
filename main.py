# -*- coding: utf-8 -*-
"""main.py  —  Marvel Rivals Classifier"""

import queue, shutil, tempfile, threading, zipfile
import multiprocessing as mp
import tkinter as tk
import tkinter.font
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from game_mode_select import is_domination
from dash_counter import count_dashes
from kill_counter import (count_kills, load_contours as load_kill_contours,
                          SLOT1_CONTOUR_PATH as BOARD_SLOT1_CONTOUR_PATH,
                          SLOT2_CONTOUR_PATH as BOARD_SLOT2_CONTOUR_PATH)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SORTED_DIR       = "final"
MAP_MODEL_PATH   = "models/map_classifier.pth"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

FRAME_INTERVAL_SECONDS = 2
IMG_SIZE = 224
DOMINATION_MAPS = {
    "Birnin TChalla", "Celestial Husk", "Hells Heaven",
    "Krakoa", "Lower Manhattan", "Royal Palace",
}
BLACKOUT_BOXES = [
    (25,   600,  900, 1060),
    (1450, 1875, 900, 1065),
    (750,  1175, 970, 1050),
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def apply_blackout(frame):
    for x0, x1, y0, y1 in BLACKOUT_BOXES:
        frame[y0:y1, x0:x1] = 0
    return frame


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    import tempfile as _tmp
    t = _tmp.NamedTemporaryFile(suffix=video_path.suffix, delete=False)
    t.close()
    shutil.copy2(str(video_path), t.name)
    return cv2.VideoCapture(t.name), t.name


def load_map_classifier(model_path: str, device):
    ckpt    = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model   = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes


def extract_pil_frames(video_path: Path) -> list:
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        return []
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * FRAME_INTERVAL_SECONDS))
    idx, frames = 0, []
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        apply_blackout(frame)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += step
    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return frames


def _kill_worker(args):
    path_str, bc1, bc2 = args
    return count_kills(Path(path_str), bc1, bc2)


def _dash_worker(path_str):
    return count_dashes(Path(path_str))


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BG      = '#0c0b14'
PANEL   = '#18162a'
BORDER  = '#2e2a4a'
PURPLE  = '#7c3aed'
PURPLE2 = '#a78bfa'
TEXT    = '#ede9fe'
DIM     = '#4c4570'
GRN     = '#34d399'
ERR     = '#f87171'
ORANGE  = '#f97316'


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class RivalsApp:

    _STAGE_META = [
        ('kills_dashes', 'Kills & Dashes'),
        ('frames',       'Frame Extract'),
        ('classify',     'Map Classify'),
        ('organize',     'Organize'),
    ]
    _STAGE_MSG = {
        'kills_dashes': 'Counting kills & dashes...',
        'frames':       'Extracting frames...',
        'classify':     'Classifying maps...',
        'organize':     'Organising clips...',
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('Marvel Rivals Classifier')
        root.configure(bg=BG)
        root.geometry('900x680')
        root.minsize(700, 500)

        self._q          = queue.Queue()
        self._selected   = []
        self._clips      = []
        self._active_map = 'all'
        self._progress   = 0
        self._chip_btns  = {}

        self._f_home = tk.Frame(root, bg=BG)
        self._f_proc = tk.Frame(root, bg=BG)
        self._f_res  = tk.Frame(root, bg=BG)

        self._build_home()
        self._build_proc()
        self._show('home')

    # ── helpers ───────────────────────────────────────────────────────────

    def _label(self, parent, text, size=9, bold=False, fg=None, bg=None):
        font = ('Segoe UI', size, 'bold' if bold else 'normal')
        return tk.Label(parent, text=text,
                        bg=bg or parent.cget('bg'),
                        fg=fg or TEXT,
                        font=font)

    def _btn(self, parent, text, cmd, fg=TEXT, bg=PANEL, hover_bg=None, hover=True, **kw):
        _hbg = hover_bg if hover_bg is not None else BORDER
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg,
                      activebackground=_hbg, activeforeground=fg,
                      disabledforeground='#5a5280',
                      relief='flat', bd=0,
                      font=('Segoe UI', 10, 'bold'),
                      padx=18, pady=10, cursor='hand2', **kw)
        if hover:
            b.bind('<Enter>', lambda e, b=b, hb=_hbg: b.config(bg=hb)
                   if str(b.cget('state')) != 'disabled' else None)
            b.bind('<Leave>', lambda e, b=b, ob=bg: b.config(bg=ob))
        return b

    def _panel(self, parent, expand=False):
        f = tk.Frame(parent, bg=PANEL, padx=20, pady=14)
        f.pack(fill='both' if expand else 'x',
               expand=expand, padx=20, pady=(0, 10))
        return f

    # ── home view ─────────────────────────────────────────────────────────

    def _build_home(self):
        root = self._f_home

        # Title
        self._label(root, 'MARVEL RIVALS CLASSIFIER',
                    size=22, bold=True, fg=PURPLE2).pack(pady=(24, 2))
        self._label(root, 'Auto-sort clips by map  ·  count kills & dashes',
                    size=9, fg=DIM).pack(pady=(0, 16))

        # File panel
        fp = self._panel(root)
        self._label(fp, 'VIDEOS', size=7, fg=DIM).pack(anchor='w')

        self._lbl_files = self._label(fp, 'No files selected', fg=DIM)
        self._lbl_files.pack(anchor='w', pady=(6, 0))

        self._lb = tk.Listbox(fp, bg=BG, fg=PURPLE2,
                              selectbackground=BG,
                              activestyle='none', relief='flat', bd=0,
                              font=('Segoe UI', 9), height=6,
                              highlightthickness=1,
                              highlightcolor=BORDER,
                              highlightbackground=BORDER)
        self._lb.pack(fill='x', pady=(6, 0))

        self._lbl_model = self._label(fp, '', size=8, fg=DIM)
        self._lbl_model.pack(anchor='w', pady=(8, 0))
        self._check_model()

        # Buttons
        row = tk.Frame(root, bg=BG)
        row.pack(fill='x', padx=20, pady=(0, 20))
        self._btn(row, 'Browse...', self._pick,
                  fg=PURPLE2, bg=BORDER, hover_bg='#3d3860').pack(side='left')
        self._btn_go = self._btn(row, 'PROCESS CLIPS', self._start,
                                 fg='#ffffff', bg=PURPLE, hover_bg='#9f7aea',
                                 state='disabled')
        self._btn_go.config(disabledforeground='#c4b5fd')
        self._btn_go.pack(side='right')

    def _check_model(self):
        if Path(MAP_MODEL_PATH).exists():
            self._lbl_model.config(text='✓  Model ready', fg=GRN)
        else:
            self._lbl_model.config(text='✗  Model not found — run train_model.py', fg=ERR)

    def _pick(self):
        paths = filedialog.askopenfilenames(
            title='Select clips',
            filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv *.webm'), ('All', '*.*')])
        if not paths:
            return
        self._selected = [Path(p) for p in paths]
        self._lb.delete(0, 'end')
        for p in self._selected:
            self._lb.insert('end', f'  {p.name}')
        n = len(self._selected)
        self._lbl_files.config(text=f'{n} file{"s" if n != 1 else ""} selected', fg=PURPLE2)
        if n > 0 and Path(MAP_MODEL_PATH).exists():
            self._btn_go.config(state='normal')

    # ── proc view ─────────────────────────────────────────────────────────

    def _build_proc(self):
        root = self._f_proc

        self._label(root, 'MARVEL RIVALS CLASSIFIER',
                    size=22, bold=True, fg=PURPLE2).pack(pady=(24, 16))

        # Progress panel
        pp = self._panel(root)
        hrow = tk.Frame(pp, bg=PANEL)
        hrow.pack(fill='x')
        self._lbl_stage = self._label(hrow, 'Starting...', fg=DIM)
        self._lbl_stage.pack(side='left')
        self._lbl_pct = self._label(hrow, '0%', size=18, bold=True, fg=PURPLE2)
        self._lbl_pct.pack(side='right')

        self._bar_cv = tk.Canvas(pp, height=6, bg=BG, highlightthickness=0)
        self._bar_cv.pack(fill='x', pady=(8, 12))

        # Stage labels
        sf = tk.Frame(pp, bg=PANEL)
        sf.pack(fill='x')
        self._stage_lbls = {}
        for i, (key, label) in enumerate(self._STAGE_META):
            lbl = self._label(sf, label, size=8, fg=DIM)
            lbl.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0))
            sf.grid_columnconfigure(i, weight=1)
            self._stage_lbls[key] = lbl

        # Log panel
        lp = self._panel(root, expand=True)
        self._label(lp, 'LOG', size=7, fg=DIM).pack(anchor='w')
        lw = tk.Frame(lp, bg=BG)
        lw.pack(fill='both', expand=True, pady=(6, 0))
        fams = tkinter.font.families()
        mono = ('Cascadia Code', 8) if 'Cascadia Code' in fams else ('Courier New', 9)
        self._log = tk.Text(lw, bg=BG, fg=DIM,
                            relief='flat', bd=0, font=mono,
                            wrap='word', state='disabled',
                            highlightthickness=0, padx=6, pady=4)
        vsb = tk.Scrollbar(lw, orient='vertical', command=self._log.yview)
        self._log.configure(yscrollcommand=vsb.set)
        self._log.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        self._log.tag_configure('ok',    foreground=GRN)
        self._log.tag_configure('warn',  foreground=ORANGE)
        self._log.tag_configure('error', foreground=ERR)

    def _draw_bar(self):
        c  = self._bar_cv
        W  = max(c.winfo_width(), 1)
        H  = c.winfo_height() or 6
        fw = int(W * self._progress / 100)
        c.delete('all')
        c.create_rectangle(0, 0, W, H, fill=BG, outline='')
        if fw > 1:
            c.create_rectangle(0, 0, fw, H, fill=PURPLE, outline='')

    def _log_write(self, text, tag=''):
        self._log.config(state='normal')
        self._log.insert('end', text + '\n', (tag,) if tag else ())
        self._log.config(state='disabled')
        self._log.see('end')

    # ── results view ──────────────────────────────────────────────────────

    def _build_results(self, clips):
        self._clips = clips or []
        for w in self._f_res.winfo_children():
            w.destroy()

        self._label(self._f_res, 'MARVEL RIVALS CLASSIFIER',
                    size=22, bold=True, fg=PURPLE2).pack(pady=(24, 16))

        rp = self._panel(self._f_res, expand=True)

        # Header
        hrow = tk.Frame(rp, bg=PANEL)
        hrow.pack(fill='x', pady=(0, 10))
        n = len(clips)
        self._label(hrow, f'{n} clip{"s" if n != 1 else ""} processed',
                    size=8, fg=DIM).pack(side='left')
        self._btn(hrow, 'Process More', self._back, fg=DIM).pack(side='right', padx=(6, 0))
        self._btn(hrow, 'Download ZIP', self._save_zip, fg=PURPLE2).pack(side='right')

        # Filter / sort bar
        frow = tk.Frame(rp, bg=PANEL)
        frow.pack(fill='x', pady=(0, 8))

        maps   = ['All'] + sorted({c['map'] for c in clips})
        combos = ['All'] + sorted({c['combos'] or 'No Combo' for c in clips})
        sorts  = ['Map A→Z', 'Kills ↓', 'Kills ↑', 'Dashes ↓', 'Dashes ↑']

        self._fv_map    = tk.StringVar(value='All')
        self._fv_combo  = tk.StringVar(value='All')
        self._fv_sort   = tk.StringVar(value='Map A→Z')
        self._fv_kills  = tk.IntVar(value=0)
        self._fv_dashes = tk.IntVar(value=0)

        def flbl(txt):
            self._label(frow, txt, size=8, fg=DIM).pack(side='left', padx=(0, 3))

        def fom(var, vals):
            m = tk.OptionMenu(frow, var, *vals, command=lambda _: self._apply_filters())
            m.config(bg=BORDER, fg=TEXT, activebackground=PURPLE, activeforeground='#ffffff',
                     relief='flat', bd=0, font=('Segoe UI', 9), padx=8, pady=6,
                     highlightthickness=0)
            m['menu'].config(bg=BG, fg=TEXT, activebackground=PURPLE,
                             activeforeground='#ffffff', font=('Segoe UI', 9), bd=0)
            m.pack(side='left', padx=(0, 10))

        def fsp(var):
            s = tk.Spinbox(frow, from_=0, to=99, width=3, textvariable=var,
                           bg=BORDER, fg=TEXT, buttonbackground=BG,
                           insertbackground=TEXT, relief='flat', bd=0,
                           font=('Segoe UI', 9), command=self._apply_filters)
            s.bind('<Return>', lambda _: self._apply_filters())
            s.pack(side='left', padx=(0, 10))

        flbl('Map');      fom(self._fv_map,   maps)
        flbl('Combo');    fom(self._fv_combo, combos)
        flbl('Min Kills'); fsp(self._fv_kills)
        flbl('Min Dashes'); fsp(self._fv_dashes)
        flbl('Sort');     fom(self._fv_sort,  sorts)

        # Clip list — scrolls only when content overflows
        cf = tk.Frame(rp, bg=PANEL)
        cf.pack(fill='both', expand=True)
        cf.grid_rowconfigure(0, weight=1)
        cf.grid_columnconfigure(0, weight=1)

        self._clips_cv = tk.Canvas(cf, bg=PANEL, highlightthickness=0)
        self._vsb = tk.Scrollbar(cf, orient='vertical', command=self._clips_cv.yview)
        self._clips_cv.configure(yscrollcommand=self._vsb_set)
        self._clips_cv.grid(row=0, column=0, sticky='nsew')
        self._vsb.grid(row=0, column=1, sticky='ns')
        self._vsb.grid_remove()

        self._clips_in = tk.Frame(self._clips_cv, bg=PANEL)
        cwin = self._clips_cv.create_window(0, 0, anchor='nw', window=self._clips_in)
        self._clips_in.bind('<Configure>',
            lambda _e: self._clips_cv.configure(
                scrollregion=self._clips_cv.bbox('all') or (0, 0, 0, 0)))
        self._clips_cv.bind('<Configure>',
            lambda e: self._clips_cv.itemconfig(cwin, width=e.width))

        self._apply_filters()
        self._show('res')

    def _vsb_set(self, lo, hi):
        self._vsb.set(lo, hi)
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self._vsb.grid_remove()
            try: self._clips_cv.unbind_all('<MouseWheel>')
            except Exception: pass
        else:
            self._vsb.grid()
            self._clips_cv.bind_all('<MouseWheel>',
                lambda e: self._clips_cv.yview_scroll(-1*(e.delta//120), 'units'))

    def _apply_filters(self):
        fmap   = self._fv_map.get()
        fcombo = self._fv_combo.get()
        min_k  = self._fv_kills.get()
        min_d  = self._fv_dashes.get()
        fsort  = self._fv_sort.get()

        out = self._clips
        if fmap != 'All':
            out = [c for c in out if c['map'] == fmap]
        if fcombo != 'All':
            target = None if fcombo == 'No Combo' else fcombo
            out = [c for c in out if (c.get('combos') or None) == target]
        out = [c for c in out if c['kills'] >= min_k and c['dashes'] >= min_d]

        key_map = {
            'Map A→Z': (lambda c: c['map'],   False),
            'Kills ↓': (lambda c: c['kills'],  True),
            'Kills ↑': (lambda c: c['kills'],  False),
            'Dashes ↓':(lambda c: c['dashes'], True),
            'Dashes ↑':(lambda c: c['dashes'], False),
        }
        if fsort in key_map:
            fn, rev = key_map[fsort]
            out = sorted(out, key=fn, reverse=rev)

        self._render_clips(out)

    def _render_clips(self, filtered):
        for w in self._clips_in.winfo_children():
            w.destroy()
        if not filtered:
            self._label(self._clips_in, 'No clips match.', fg=DIM).pack(
                pady=12, anchor='nw')
            return
        for clip in filtered:
            row = tk.Frame(self._clips_in, bg=BG, pady=8, padx=10)
            row.pack(fill='x', pady=(0, 4), anchor='nw')
            self._btn(row, 'Save',
                      lambda fn=clip['filename']: self._save_clip(fn),
                      fg=PURPLE2, bg=PANEL).pack(side='right')
            info = tk.Frame(row, bg=BG)
            info.pack(side='left', fill='x', expand=True)
            stats = f'[{clip["map"]}]  Kills {clip["kills"]}  Dashes {clip["dashes"]}'
            if clip.get('combos'):
                stats += f'  ·  {clip["combos"]}'
            stats += f'  ({clip["conf"]}%)'
            self._label(info, stats, size=8, fg=PURPLE2, bg=BG).pack(anchor='w')
            self._label(info, clip['filename'], size=8, fg=TEXT, bg=BG).pack(anchor='w')

    def _save_clip(self, filename):
        src = Path(SORTED_DIR) / filename
        if not src.exists():
            messagebox.showerror('Not found', f'{filename} not found.')
            return
        dst = filedialog.asksaveasfilename(
            defaultextension=src.suffix,
            filetypes=[('Video', f'*{src.suffix}'), ('All', '*.*')],
            initialfile=filename)
        if dst:
            shutil.copy2(str(src), dst)

    def _save_zip(self):
        files = [Path(SORTED_DIR) / c['filename'] for c in self._clips]
        files = [f for f in files if f.exists()]
        if not files:
            messagebox.showwarning('Empty', 'No clips from this session to zip.')
            return
        dst = filedialog.asksaveasfilename(
            defaultextension='.zip',
            filetypes=[('ZIP', '*.zip')],
            initialfile='rivals_clips.zip')
        if not dst:
            return
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_STORED) as zf:
            for f in files:
                zf.write(f, f.name)

    def _back(self):
        self._show('home')
        self._progress = 0
        self._draw_bar()
        self._lbl_pct.config(text='0%', fg=PURPLE2)
        self._lbl_stage.config(text='Starting...', fg=DIM)
        self._log.config(state='normal')
        self._log.delete('1.0', 'end')
        self._log.config(state='disabled')
        for lbl in self._stage_lbls.values():
            lbl.config(fg=DIM)

    # ── view switching ─────────────────────────────────────────────────────

    def _show(self, name):
        for n, f in [('home', self._f_home),
                     ('proc', self._f_proc),
                     ('res',  self._f_res)]:
            if n == name:
                f.pack(fill='both', expand=True)
            else:
                f.pack_forget()

    # ── pipeline ──────────────────────────────────────────────────────────

    def _start(self):
        self._btn_go.config(state='disabled')
        self._show('proc')
        threading.Thread(target=self._pipeline, daemon=True).start()
        self._poll()

    def _emit(self, t, **kw):
        self._q.put({'type': t, **kw})

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                self._handle(msg)
                if msg['type'] in ('done', 'error'):
                    return
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _handle(self, msg):
        t = msg['type']
        if t == 'progress':
            self._progress = msg['value']
            self._lbl_pct.config(text=f"{msg['value']}%")
            self._draw_bar()
        elif t == 'stage':
            key, status = msg['stage'], msg['status']
            lbl = self._stage_lbls.get(key)
            if lbl:
                if status == 'active':
                    lbl.config(fg=PURPLE2)
                    self._lbl_stage.config(
                        text=self._STAGE_MSG.get(key, key), fg=DIM)
                elif status == 'done':
                    lbl.config(fg=GRN)
        elif t == 'log':
            self._log_write(msg['message'], msg.get('tag', ''))
        elif t == 'done':
            self._progress = 100
            self._draw_bar()
            self._lbl_pct.config(text='100%', fg=GRN)
            self._lbl_stage.config(text='Done!', fg=GRN)
            self.root.after(800, lambda: self._build_results(msg['results']))
        elif t == 'error':
            self._log_write(msg['message'], 'error')
            self._lbl_stage.config(text='Error', fg=ERR)

    def _pipeline(self):
        try:
            videos = self._selected
            if not videos:
                self._emit('error', message='No videos selected.')
                return
            if not Path(MAP_MODEL_PATH).exists():
                self._emit('error', message=f'Model not found at {MAP_MODEL_PATH}.')
                return

            nw = max(1, mp.cpu_count() - 1)
            self._emit('log', message=f'Processing {len(videos)} video(s)  |  workers: {nw}')
            self._emit('stage', stage='kills_dashes', status='active')
            self._emit('progress', value=5)

            bc1 = load_kill_contours(BOARD_SLOT1_CONTOUR_PATH)
            bc2 = load_kill_contours(BOARD_SLOT2_CONTOUR_PATH)

            with mp.Pool(processes=nw) as pool:
                kf = pool.map_async(_kill_worker, [(str(v), bc1, bc2) for v in videos])
                df = pool.map_async(_dash_worker, [str(v) for v in videos])
                kill_results = kf.get()
                dash_results = df.get()

            kills_map  = {nm: tot for nm, tot, _ in kill_results}
            dashes_map = {nm: (tot, cb) for nm, tot, _, cb, __ in dash_results}

            for nm, tot, _ in kill_results:
                self._emit('log', message=f'  [{nm}]  kills {tot}')
            for nm, tot, _, cb, __ in dash_results:
                cs = ('  (' + ' - '.join(lbl for _, lbl in cb) + ')') if cb else ''
                self._emit('log', message=f'  [{nm}]  dashes {tot}{cs}')

            self._emit('progress', value=40)
            self._emit('stage', stage='kills_dashes', status='done')

            self._emit('stage', stage='frames', status='active')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._emit('log', message=f'Device: {device}')
            classifier, classes = load_map_classifier(MAP_MODEL_PATH, device)

            with ThreadPoolExecutor(max_workers=nw) as ex:
                video_data = list(ex.map(
                    lambda vp: (extract_pil_frames(vp), is_domination(vp)[0]),
                    videos))

            self._emit('progress', value=65)
            self._emit('stage', stage='frames', status='done')

            self._emit('stage', stage='classify', status='active')
            all_tensors, frame_counts, valid_idx = [], [], []
            for i, (pil_frames, _) in enumerate(video_data):
                if not pil_frames:
                    frame_counts.append(0)
                    continue
                ts = [_transform(pf) for pf in pil_frames]
                all_tensors.extend(ts)
                frame_counts.append(len(ts))
                valid_idx.append(i)

            map_by_name = {}
            if all_tensors:
                batch = torch.stack(all_tensors).to(device)
                with torch.no_grad():
                    all_probs = F.softmax(classifier(batch), dim=1).cpu().numpy()
                ptr = 0
                for i in valid_idx:
                    cnt  = frame_counts[i]
                    avg  = np.mean(all_probs[ptr:ptr + cnt], axis=0)
                    ptr += cnt
                    _, dom = video_data[i]
                    allowed = (DOMINATION_MAPS if dom
                               else {c for c in classes if c not in DOMINATION_MAPS})
                    mask = np.array([c in allowed for c in classes], dtype=float)
                    avg  = avg * mask
                    best = int(np.argmax(avg))
                    map_by_name[videos[i].name] = (classes[best], float(avg[best]))

            self._emit('progress', value=80)
            self._emit('stage', stage='classify', status='done')

            self._emit('stage', stage='organize', status='active')
            Path(SORTED_DIR).mkdir(parents=True, exist_ok=True)
            results = []
            n = len(videos)
            for idx, vp in enumerate(videos):
                nm = vp.name
                if nm not in map_by_name:
                    self._emit('log', message=f'  SKIP {nm}')
                    continue
                map_name, conf = map_by_name[nm]
                kills          = kills_map.get(nm, 0)
                dashes, combos = dashes_map.get(nm, (0, []))
                combo_str = (' - '.join(lbl for _, lbl in combos)) if combos else None
                stem = (f'{map_name} - {dashes}d - {combo_str} - {kills}k'
                        if combo_str else f'{map_name} - {dashes}d - {kills}k')
                dest = Path(SORTED_DIR) / f'{stem}{vp.suffix}'
                c2 = 1
                while dest.exists():
                    dest = Path(SORTED_DIR) / f'{stem} ({c2}){vp.suffix}'
                    c2 += 1
                shutil.copy2(str(vp), str(dest))
                results.append({'original': nm, 'filename': dest.name,
                                'map': map_name, 'conf': round(conf * 100),
                                'kills': kills, 'dashes': dashes, 'combos': combo_str})
                self._emit('log', message=f'  {dest.name}', tag='ok')
                self._emit('progress', value=80 + int(20 * (idx + 1) / n))

            self._emit('stage', stage='organize', status='done')
            self._emit('progress', value=100)
            self._emit('done', results=results)

        except Exception as exc:
            import traceback
            self._emit('error', message=f'{exc}\n{traceback.format_exc()}')


# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    RivalsApp(root)
    root.mainloop()


if __name__ == '__main__':
    mp.freeze_support()
    main()
