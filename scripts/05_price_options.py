"""Price the wide decision (Table 1): each option against recycling, per 100 five-second decisions, comparing decisions
where the option was equally likely (graph attention propensities, cross-fitted gradient-boosting outcome models,
doubly robust estimation, match-clustered bootstrap). Also the choice shares and, descriptively, where the ball is lost.

Writes results/table1.json.

Usage: python scripts/05_price_options.py --out derived [--propensities derived/propensities.parquet]
"""
import os

# one thread keeps the gradient-boosting fits identical from run to run
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowblock import config as C  # noqa: E402
from lowblock.aipw import folds_by_match, price  # noqa: E402
from lowblock.choice_data import CONTEXT_GBM  # noqa: E402
from lowblock.outcome_models import OUTCOMES, crossfit_choice, crossfit_outcome  # noqa: E402
from lowblock.prep import prepare  # noqa: E402

KEYS = ["season", "match_id", "team_id", "possession", "k"]
ROWS = [("pass or carry into the block", "A", 1), ("cross", "A", 2), ("lofted cross", "B", 2), ("low cross", "B", 3)]
LOSS_GROUPS = ["cut-back", "lofted cross", "low cross", "pass or carry into the block", "recycle"]


def where_lost(S, n_boot=C.N_BOOT, seed=C.SEED_LOSS):
    """Median distance from the attacking team's own goal of attack-ending ball losses (no shot in the attack's last 5 s)
    whose last step is a wide decision with the block still low; grouped by the option in that step's window."""
    key = ["season", "match_id", "possession"]
    S = S.sort_values(key + ["k"])
    S["k_last"] = S.groupby(key).k.transform("max")
    last = S.groupby(key).tail(1)
    r5 = S[S.k == np.maximum(S.k_last - 4, 0)][key + ["rest_shot"]].rename(columns={"rest_shot": "shots_last5"})
    last = last.merge(r5, on=key, how="left")
    L = last[(last.next_team == last.opp_id) & (last.shots_last5 == 0) & last.wide & last.low_dec].copy()
    into = (L.pass_into_block + L.carry_into_block) > 0
    L["group"] = np.select([L.ctype == "cutback", L.ctype == "cross_high", L.ctype == "cross_low", into], LOSS_GROUPS[:4],
                           LOSS_GROUPS[4])
    rng = np.random.default_rng(seed)
    out = {}
    for lab, m in [(g, L.group == g) for g in LOSS_GROUPS] + [("cross", L.cross.astype(bool))]:
        for s in ["pooled"] + C.SEASONS:
            D = L[m] if s == "pooled" else L[m & (L.season == s)]
            if not len(D):
                continue
            mids = D.match_id.unique()
            grp = {k: v.loss_dist.to_numpy() for k, v in D.groupby("match_id")}
            bs = [np.nanmedian(np.concatenate([grp[k] for k in rng.choice(mids, len(mids), replace=True)]))
                  for _ in range(n_boot if s == "pooled" else C.N_BOOT_SEASON)]
            out.setdefault(lab, {})[s] = {"losses": int(len(D)), "median_m_from_own_goal": [
                float(np.nanmedian(D.loss_dist)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder for the JSON results")
    ap.add_argument("--propensities", help="defaults to --out/propensities.parquet")
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    out = Path(a.out)
    U = pd.read_parquet(out / "choice_table.parquet")
    U = U[U.low_dec].reset_index(drop=True)
    feats = [c for c in U.columns if c.startswith("f_")] + CONTEXT_GBM
    X = U[feats].to_numpy(np.float32)
    fold = folds_by_match(U.match_id.to_numpy(), C.N_FOLDS, C.SEED)
    groups = U.match_id.to_numpy()
    Y = {o: U[o].to_numpy(float) for o, _ in OUTCOMES}
    nuis = {}
    for sc, col, K in (("A", "choice", 3), ("B", "choice_type", 5)):
        Asc = U[col].to_numpy(int)
        nuis[sc] = {"A": Asc, "Mu": {o: crossfit_outcome(X, Asc, Y[o], K, fold, kind) for o, kind in OUTCOMES}}
        print(f"outcome models scheme {sc} done", flush=True)
    # splits the graph model's cross probability across cross types
    P_type = crossfit_choice(X, nuis["B"]["A"], 5, fold)
    G = pd.read_parquet(a.propensities or out / "propensities.parquet")
    G["season"] = G.season.astype(str)
    g = U[KEYS].assign(season=U.season.astype(str)).merge(G, on=KEYS, how="left")
    eA = g[["p0", "p1", "p2"]].to_numpy(float)
    assert np.isfinite(eA).all(), "missing propensities"
    share = P_type[:, 2:5] / np.clip(P_type[:, 2:5].sum(1, keepdims=True), 1e-12, None)
    e = {"A": eA, "B": np.column_stack([eA[:, 0], eA[:, 1], eA[:, 2:3] * share])}
    res = {"n_decisions": int(len(U)), "units": "per 100 five-second wide decisions vs recycling; [estimate, 2.5%, 97.5%]",
           "options": {}}
    for lab, sc, arm in ROWS:
        r = price(e[sc], nuis[sc]["Mu"], nuis[sc]["A"], Y, arm, groups, trim=C.TRIM, per=100, n_boot=a.boot, seed=C.SEED)
        res["options"][lab] = {"extra_balls_lost_5s": r["turn5"], "counter_xg_20s": r["opp_xg20"],
                               "xg_created_rest_of_attack": r["rest_xg"], "net_xg": r["net_xg"], "n_kept": r["n_kept"]}
    S = prepare(pd.read_parquet(out / "steps.parquet"), pd.read_parquet(out / "attacks.parquet"),
                pd.read_parquet(out / "strength.parquet"))
    res["where_lost"] = where_lost(S)
    W = S[S.wide & S.low_dec]                     # every wide moment with the block still low
    res["shares"] = {"n_moments": int(len(W)),
                     "definition": "cross = any cross type in the window (lofted, low or cut-back); recycle = the rest",
                     "pct": {"pass or carry into the block": 100 * float((W.choice == 1).mean()),
                             "cross": 100 * float((W.ctype != "").mean()),
                             "lofted cross": 100 * float((W.ctype == "cross_high").mean()),
                             "low cross": 100 * float((W.ctype == "cross_low").mean()),
                             "cut-back": 100 * float((W.ctype == "cutback").mean())}}
    res["shares"]["pct"]["recycle"] = 100 - res["shares"]["pct"]["pass or carry into the block"] - res["shares"]["pct"]["cross"]
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "table1.json").write_text(json.dumps(res, indent=1))
    f = lambda v, d=2: f"{v[0]:+.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"
    print(f"shares % ({res['shares']['n_moments']} wide moments):", {k: round(v, 1) for k, v in res["shares"]["pct"].items()})
    for lab, r in res["options"].items():
        print(f"{lab}: balls lost {f(r['extra_balls_lost_5s'], 1)} | counter xG {f(r['counter_xg_20s'])} | "
              f"xG created {f(r['xg_created_rest_of_attack'])} | net {f(r['net_xg'])}")
    print("where lost (median m from own goal):", {k: round(v["pooled"]["median_m_from_own_goal"][0], 1) for k, v in res["where_lost"].items()})


if __name__ == "__main__":
    main()
