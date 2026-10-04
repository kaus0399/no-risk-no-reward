"""Inputs for the choice models at the decision moment (t-5 s).

Nodes: 23 x (depth from the goal line being attacked, lateral, vx, vy); slots 0-10 attackers (goalkeeper first), 11-21
defenders (goalkeeper first), 22 the ball. Every decision is mirrored so the ball's lateral position is >= 0.
"""
import numpy as np
import pandas as pd

from . import config as C

CONTEXT = ["minute", "score_diff", "att_str", "def_str", "regain", "throw_in_f", "since_set"]
CONTEXT_GBM = ["minute", "score_diff", "att_str", "def_str", "season_2", "regain", "throw_in_f", "since_set"]
NEAREST = 6


def add_context(S, strength):
    st = strength.set_index(["match_id", "team_id"]).npxgd_pm
    S = S.copy()
    S["att_str"] = pd.MultiIndex.from_arrays([S.match_id, S.team_id]).map(st).to_numpy(float)
    S["def_str"] = pd.MultiIndex.from_arrays([S.match_id, S.opp_id]).map(st).to_numpy(float)
    S["season_2"] = (S.season == C.SEASONS[1]).astype(float)
    S["regain"] = (S.start == "regain").astype(float)
    S["throw_in_f"] = S.throw_in.astype(float)
    S["since_set"] = S.k - 5.0
    return S


def mirrored(W, M):
    """Mirror [n, 23, 4] nodes so the ball's lateral position at the decision moment is >= 0."""
    W = W.copy()
    flip = (W[:, 22, 1] < 0) & M[:, 22]
    W[flip, :, 1] *= -1
    W[flip, :, 3] *= -1
    return W


def two_lines_vec(depth_gs, n_gs, min_line=C.MIN_LINE):
    """Back and midfield line depth for each row from the goal-side defenders' depths (inf padded)."""
    n, m = depth_gs.shape
    g = np.sort(depth_gs, axis=1)
    gaps = np.diff(g, axis=1)
    k = np.arange(m - 1)[None, :]
    ok = (k >= min_line - 1) & (k <= (n_gs[:, None] - min_line - 1))
    gaps = np.where(ok & np.isfinite(gaps), gaps, -np.inf)
    ks = gaps.argmax(1)
    valid = n_gs >= 2 * min_line
    gf = np.where(np.isfinite(g), g, 0.0)
    cs = np.cumsum(gf, axis=1)
    back = cs[np.arange(n), ks] / (ks + 1)
    tot = cs[np.arange(n), np.clip(n_gs - 1, 0, m - 1)]
    mid = (tot - cs[np.arange(n), ks]) / np.maximum(n_gs - ks - 1, 1)
    back[~valid] = np.nan
    mid[~valid] = np.nan
    return back, mid


def tabular_features(F, Mk):
    """Hand-built features at the decision moment for the gradient-boosting models. F: [n, 23, 4], Mk: [n, 23]."""
    F = F.astype(np.float32).copy()
    F[~Mk] = np.nan
    flip = F[:, 22, 1] < 0
    F[flip, :, 1] *= -1
    F[flip, :, 3] *= -1
    bx, by, bvx, bvy = (F[:, 22, j] for j in range(4))
    out = {"f_ball_depth": bx, "f_ball_y": by, "f_ball_vx": bvx, "f_ball_vy": bvy, "f_ball_speed": np.hypot(bvx, bvy)}
    n = len(F)
    ar = np.arange(n)[:, None]
    nearest = {}
    for side, P in (("a", F[:, 0:11]), ("d", F[:, 11:22])):
        dx, dy = P[..., 0] - bx[:, None], P[..., 1] - by[:, None]
        d = np.hypot(dx, dy)
        o = np.argsort(np.where(np.isnan(d), np.inf, d), axis=1)[:, :NEAREST]
        nearest[side] = o
        for nm, arr in (("dx", dx), ("dy", dy), ("d", d), ("vx", P[..., 2]), ("vy", P[..., 3])):
            v = arr[ar, o]
            for i in range(NEAREST):
                out[f"f_{side}{i + 1}_{nm}"] = v[:, i]
        out[f"f_{side}_n10"] = (d <= 10).sum(1).astype(np.float32)
        if side == "d":
            out["f_d_n5"] = (d <= 5).sum(1).astype(np.float32)
    out["f_overload10"] = out["f_a_n10"] - out["f_d_n10"]
    A, D = F[:, 0:11], F[:, 12:22]
    inb = lambda Q: (Q[..., 0] <= 16.5) & (np.abs(Q[..., 1]) <= C.BOX_HALF_WIDTH)
    out["f_att_box"] = inb(A).sum(1).astype(np.float32)
    out["f_def_box"] = inb(D).sum(1).astype(np.float32)
    gs = np.hypot(D[..., 0], D[..., 1]) < np.hypot(bx, by)[:, None]
    n_gs = gs.sum(1)
    back, mid = two_lines_vec(np.where(gs, D[..., 0], np.inf), n_gs)
    out.update({"f_n_goalside": n_gs.astype(np.float32), "f_back_depth": back, "f_mid_depth": mid, "f_line_gap": mid - back})
    dd = np.sort(np.where(np.isnan(D[..., 0]), np.inf, D[..., 0]), axis=1)[:, :3]
    out["f_deep3"] = np.where(np.isfinite(dd).all(1), dd.mean(1), np.nan)
    c = nearest["a"][:, 0]
    pc = A[np.arange(n), c]
    rel = F[:, 11:22, :2] - pc[:, None, :2]
    relv = F[:, 11:22, 2:] - pc[:, None, 2:]
    dc = np.hypot(rel[..., 0], rel[..., 1])
    j = np.argmin(np.where(np.isnan(dc), np.inf, dc), axis=1)
    dmin = dc[np.arange(n), j]
    out["f_press_d"] = dmin
    out["f_press_close"] = -(rel[np.arange(n), j] * relv[np.arange(n), j]).sum(1) / np.maximum(dmin, 0.1)
    return pd.DataFrame(out)
