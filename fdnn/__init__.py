"""fdnn — neural-network dash detector (vendored from C:/Adhitya/Coding/test/FDNN).

A two-stage detector: a frozen ImageNet ResNet18 turns every frame into a 512-d
feature vector (``features.py``), then the SDSNN temporal head (``dnn.py``) emits
a per-frame "a dash just completed" probability. 1-D non-max suppression over
that track (``peaks.py``) collapses each bump to one counted dash.

``counter.count_dashes_fdnn(video_path, threshold)`` returns the SAME 5-tuple as
the classical contour ``dash_counter.count_dashes`` so the two are drop-in
interchangeable in the pipeline.
"""
