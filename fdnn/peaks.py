"""peaks.py — turn the SDSNN per-frame probability track into dash COUNTS.

Copied from C:/Adhitya/Coding/test/FDNN/dash_code/peaks.py (the nms_peaks
decoder). The completion reframe makes #peaks == #dashes: the model emits a
smooth bump at each dash's falling edge, and 1-D non-max suppression collapses
each bump to one counted dash. NMS (not run-length grouping) is what separates
CHAINED dashes whose bumps sit a few frames apart.
"""

import numpy as np

from . import config as C


def nms_peaks(prob, threshold=None, min_dist=None):
    """1-D non-max suppression. Greedily take the highest frame >= threshold,
    record it, suppress everything within +/- min_dist, repeat. Returns sorted
    peak frame indices (= the counted dashes)."""
    threshold = C.PEAK_THRESHOLD if threshold is None else threshold
    min_dist  = C.PEAK_MIN_DIST  if min_dist  is None else min_dist
    prob = np.asarray(prob)
    cand = np.where(prob >= threshold)[0]
    if cand.size == 0:
        return []
    order = cand[np.argsort(-prob[cand])]
    taken = np.zeros(len(prob), dtype=bool)
    chosen = []
    for i in order:
        lo, hi = max(0, i - min_dist), min(len(prob), i + min_dist + 1)
        if taken[lo:hi].any():
            continue
        chosen.append(int(i)); taken[i] = True
    return sorted(chosen)
