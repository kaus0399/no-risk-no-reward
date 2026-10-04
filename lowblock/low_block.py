"""Set-low-block geometry: the two defensive lines, the moment a block becomes set, and whether the block is still
low at the decision moment."""
import numpy as np
import pandas as pd

from . import config as C


def first_set(cond):
    """Index of the frame at which `cond` (boolean, 5 Hz) has held for 3 s, breaks shorter than 1 s ignored; -1 if
    never. The returned frame and the frame that starts the 3 s run must both satisfy the condition."""
    n = len(cond)
    if n < C.SET_FRAMES or not cond.any():
        return -1
    f = cond.copy()
    d = np.diff(np.r_[0, (~cond).astype(np.int8), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s < C.FLICKER_FRAMES and s > 0 and e < n:
            f[s:e] = True
    cs = np.r_[0, np.cumsum(f.astype(np.int32))]
    win = cs[C.SET_FRAMES:] - cs[:-C.SET_FRAMES]
    ok = (win == C.SET_FRAMES) & cond[: n - C.SET_FRAMES + 1] & cond[C.SET_FRAMES - 1:]
    return int(np.argmax(ok)) + C.SET_FRAMES - 1 if ok.any() else -1


def two_lines(depth):
    """Split defenders into a back line and a midfield line at the biggest gap in depth, with at least three players per
    line. Returns (back depth, midfield depth, back mask) or None."""
    n = len(depth)
    if n < 2 * C.MIN_LINE:
        return None
    o = np.argsort(depth)
    gaps = np.diff(depth[o])
    ks = np.arange(C.MIN_LINE - 1, n - C.MIN_LINE)
    k = ks[np.argmax(gaps[ks])]
    back = np.zeros(n, bool)
    back[o[:k + 1]] = True
    return float(depth[back].mean()), float(depth[~back].mean()), back


def low_at_decision(steps):
    """Block still low at the decision moment: deepest three within 25 m at the attack's step max(k-5, 0) (nearest
    earlier step if that one is missing). Expects columns match_id, possession, period, k, deep3_nodes_t."""
    f = steps[["match_id", "possession", "period", "k", "deep3_nodes_t"]].copy()
    f["_r"] = np.arange(len(f))
    f["low_t"] = f.deep3_nodes_t <= C.DEEP3_MAX
    dec = f[["match_id", "possession", "period", "k", "low_t"]].rename(columns={"k": "k_dec", "low_t": "low_dec"})
    f["k_dec"] = (f.k - 5).clip(lower=0)
    f = pd.merge_asof(f.sort_values("k_dec"), dec.sort_values("k_dec"), on="k_dec",
                      by=["match_id", "possession", "period"], direction="backward").sort_values("_r")
    return f.low_t.to_numpy(bool), f.low_dec.fillna(False).to_numpy(bool)
