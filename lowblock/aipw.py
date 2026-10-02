"""Doubly robust (AIPW) comparison of options at equal likelihood, with a match-clustered bootstrap.

Inputs, one row per decision:
  e      [n, K] out-of-season choice probabilities from the graph attention model (column 0 = recycle)
  mu     {outcome: [n, K]} cross-fitted outcome predictions under each option (gradient boosting, 5 folds by match)
  A      [n] option actually taken (0 = recycle)
  Y      {outcome: [n]} observed outcomes, e.g. xG over the rest of the attack, counter xG within 20 s,
         ball lost within 5 s
  groups [n] match identifiers, for clustering

Rows where the option or recycling is very unlikely (probability below `trim`) are left out, so every comparison is
made where both choices were realistic.
"""
import numpy as np
import pandas as pd


def folds_by_match(match_ids, n_folds=5, seed=0):
    """Assign whole matches to cross-fitting folds."""
    u = np.unique(match_ids)
    perm = np.random.default_rng(seed).permutation(u)
    return pd.Series(np.arange(len(perm)) % n_folds, index=perm).reindex(match_ids).to_numpy()


def aipw_scores(e, mu, A, Y, arm):
    """Per-decision AIPW estimate of the mean outcome had everyone taken `arm`."""
    return mu[:, arm] + (A == arm) * (Y - mu[:, arm]) / np.maximum(e[:, arm], 1e-9)


def cluster_bootstrap(cols, groups, n_boot=1000, seed=0):
    """Point means and match-clustered bootstrap means for each column in `cols` (dict of arrays)."""
    names = list(cols)
    V = np.column_stack([cols[c] for c in names])
    gi, gu = pd.factorize(groups)
    S = np.zeros((len(gu), V.shape[1]))
    np.add.at(S, gi, V)
    N = np.bincount(gi, minlength=len(gu)).astype(float)
    rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu)) for _ in range(n_boot)]).astype(float)
    boot = (C @ S) / (C @ N)[:, None]
    point = S.sum(0) / N.sum()
    return dict(zip(names, point)), {c: boot[:, i] for i, c in enumerate(names)}


def price(e, mu, A, Y, arm, groups, trim=0.02, per=100, n_boot=1000, seed=0):
    """Difference in each outcome between taking `arm` and recycling (arm 0), per `per` decisions, with 95% intervals.

    Also returns the net value: xG created minus counter xG conceded, if both "rest_xg" and "opp_xg20" are given.
    """
    keep = (e[:, arm] >= trim) & (e[:, 0] >= trim)
    cols = {}
    for o in Y:
        cols[f"{o}|a"] = aipw_scores(e, mu[o], A, Y[o], arm)[keep]
        cols[f"{o}|0"] = aipw_scores(e, mu[o], A, Y[o], 0)[keep]
    point, boot = cluster_bootstrap(cols, groups[keep], n_boot, seed)
    out = {}
    for o in Y:
        d_pt = point[f"{o}|a"] - point[f"{o}|0"]
        d_bs = boot[f"{o}|a"] - boot[f"{o}|0"]
        out[o] = [per * d_pt, per * np.percentile(d_bs, 2.5), per * np.percentile(d_bs, 97.5)]
    if "rest_xg" in Y and "opp_xg20" in Y:
        n_pt = (point["rest_xg|a"] - point["opp_xg20|a"]) - (point["rest_xg|0"] - point["opp_xg20|0"])
        n_bs = (boot["rest_xg|a"] - boot["opp_xg20|a"]) - (boot["rest_xg|0"] - boot["opp_xg20|0"])
        out["net_xg"] = [per * n_pt, per * np.percentile(n_bs, 2.5), per * np.percentile(n_bs, 97.5)]
    out["n_kept"] = int(keep.sum())
    return out
