"""Adjusted comparisons of each option against not taking it (Figure 1b): logit for the turnover odds ratio, Poisson
for xG rate ratios, OLS for the net xG difference; match-clustered sandwich standard errors.

Adjustment: zone of the ball at the decision moment, attacking and defending team strength thirds, score, minute,
seconds since the block set, regain start, throw-in start, season."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .prep import MIN_LAB, TER, TSS_LAB


def controls(S):
    cols = []
    for lv in S.zone.unique():
        if lv != "25-35|wide":
            cols.append((S.zone == lv).to_numpy(float))
    for c, ref, levels in (("att_ter", "middle", TER), ("def_ter", "middle", TER),
                           ("score", "drawing", ["losing", "drawing", "winning"]), ("minb", "0-30", MIN_LAB)):
        for lv in levels:
            if lv != ref:
                cols.append((S[c] == lv).to_numpy(float))
    cols.append((S.start == "regain").to_numpy(float))
    cols.append(S.throw_in.to_numpy(float))
    for lv in TSS_LAB[1:]:
        cols.append((S.tss == lv).to_numpy(float))
    if S.season.nunique() > 1:
        cols.append((S.season == sorted(S.season.unique())[-1]).to_numpy(float))
    return cols


def _cluster_meat(scores, groups):
    gi = pd.factorize(groups)[0]
    Sg = np.zeros((gi.max() + 1, scores.shape[1]))
    np.add.at(Sg, gi, scores)
    return Sg, Sg.shape[0]


def poisson_cluster(X, y, groups, iters=100):
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(y.mean(), 1e-9))
    for _ in range(iters):
        eta = X @ b
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        bn = np.linalg.solve(XtW @ X + 1e-10 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn
            break
        b = bn
    mu = np.exp(X @ b)
    bread = np.linalg.inv((X.T * mu) @ X)
    Sg, G = _cluster_meat(X * (y - mu)[:, None], groups)
    return b, G / (G - 1) * bread @ (Sg.T @ Sg) @ bread


def logit_cluster(X, y, groups, iters=60):
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    for _ in range(iters):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-9)
        z = eta + (y - p) / w
        XtW = X.T * w
        bn = np.linalg.solve(XtW @ X + 1e-10 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn
            break
        b = bn
    p = 1 / (1 + np.exp(-(X @ b)))
    bread = np.linalg.pinv((X.T * (p * (1 - p))) @ X)
    Sg, G = _cluster_meat(X * (y - p)[:, None], groups)
    return b, G / (G - 1) * bread @ (Sg.T @ Sg) @ bread


def ols_cluster(X, y, groups):
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    Sg, G = _cluster_meat(X * (y - X @ b)[:, None], groups)
    return b, G / (G - 1) * XtX @ (Sg.T @ Sg) @ XtX


def ratio(b, V, j):
    se = np.sqrt(V[j, j])
    return {"rr": float(np.exp(b[j])), "lo": float(np.exp(b[j] - 1.96 * se)), "hi": float(np.exp(b[j] + 1.96 * se)),
            "p": float(2 * (1 - norm.cdf(abs(b[j] / se))))}


def fit(S, terms, ycol, kind):
    """Coefficients for `terms` with the standard adjustment. kind: 'logit' (odds ratio), 'pois' (rate ratio),
    'ols' (difference)."""
    X = np.column_stack([np.ones(len(S))] + [S[t].to_numpy(float) for t in terms] + controls(S))
    keep = X.std(0) > 0
    keep[0] = True
    X = X[:, keep]
    idx = np.cumsum(keep) - 1
    y = S[ycol].to_numpy(float)
    g = S.match_id.to_numpy()
    b, V = {"logit": logit_cluster, "pois": poisson_cluster, "ols": ols_cluster}[kind](X, y, g)
    out = {}
    for i, t in enumerate(terms):
        if not keep[1 + i]:
            continue
        j = idx[1 + i]
        if kind == "ols":
            se = np.sqrt(V[j, j])
            out[t] = {"d": float(b[j]), "lo": float(b[j] - 1.96 * se), "hi": float(b[j] + 1.96 * se)}
        else:
            out[t] = ratio(b, V, j)
    return out


def option_ratios(S, terms, reference_mask):
    """Risk (turnover odds ratio) and reward (net xG ratio) of `terms` fitted together, against not taking them.
    Net ratio = 1 + adjusted net xG difference / mean net xG of the reference rows."""
    base = float(S[reference_mask].net_xg.mean())
    nd = fit(S, terms, "net_xg", "ols")
    res = {"turnover_or": fit(S, terms, "turn5", "logit"), "xg_rr": fit(S, terms, "rest_xg", "pois"),
           "counter_xg_rr": fit(S, terms, "opp_xg20", "pois"), "net_diff": nd, "base_net": base}
    res["net_ratio"] = {t: {"rr": 1 + v["d"] / base, "lo": 1 + v["lo"] / base, "hi": 1 + v["hi"] / base}
                        for t, v in nd.items()}
    return res
