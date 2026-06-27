# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Marvel Rivals Classifier desktop app.

Build with:   pyinstaller MarvelRivalsClassifier.spec  --noconfirm
Output:       dist/MarvelRivalsClassifier/MarvelRivalsClassifier.exe  (double-click)

One-folder build (not one-file) because torch is ~4.7 GB — a one-file exe would
re-extract gigabytes to a temp dir on every launch. The whole dist folder is the
distributable; the .exe inside it is what you double-click.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Bundle the web UI and the models alongside the app.
datas = [
    ('web', 'web'),
    ('models', 'models'),
]
binaries = []
hiddenimports = [
    'clr',                       # pythonnet — pywebview edgechromium backend
    'pipeline', 'kill_counter', 'dash_counter', 'game_mode_select',
    # FDNN neural dash detector (imported lazily inside pipeline)
    'fdnn', 'fdnn.counter', 'fdnn.features', 'fdnn.dnn',
    'fdnn.peaks', 'fdnn.config',
]

# Pull in everything the heavy native packages need.
for pkg in ('torch', 'torchvision', 'cv2', 'PIL', 'numpy', 'webview'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # pragma: no cover
        print(f'[spec] collect_all({pkg!r}) skipped: {exc}')

hiddenimports += collect_submodules('webview')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pytest', 'IPython', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MarvelRivalsClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                 # no console window — clean desktop app
    disable_windowed_traceback=False,
    icon='web/icon.ico' if __import__('os').path.exists('web/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MarvelRivalsClassifier',
)
