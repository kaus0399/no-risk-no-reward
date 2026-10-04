"""Train the graph attention choice model with a season swap (train on one season, predict the other), and compare its
held-out cross AUC with a model that only sees the ball's position.

Writes --out/propensities.parquet and results/choice_model.json.

Usage: python scripts/04_train_choice_model.py --out derived [--device mps|cuda|cpu]
       python scripts/04_train_choice_model.py --out derived --propensities FILE   (evaluate saved propensities only)
"""
import os

# two CPU threads; the graph model itself runs on the GPU when one is available
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowblock import config as C  # noqa: E402
from lowblock.choice_model import train_season_swap, validation_matches  # noqa: E402

KEYS = ["season", "match_id", "team_id", "possession", "k"]


def ball_only_auc(T, A, val_matches, n_boot=C.N_BOOT, seed=C.SEED_AUC):
    """Cross AUC on the second season for the graph model and a ball-position-only model trained on the same rows."""
    mids = T.match_id.to_numpy()
    tr_s, te = (T.season == C.SEASONS[0]).to_numpy(), (T.season == C.SEASONS[1]).to_numpy()
    is_va = np.isin(mids, list(val_matches))
    itr, iva = tr_s & ~is_va, tr_s & is_va
    X = T[["f_ball_depth", "f_ball_y"]].to_numpy(np.float32)
    m = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=400, min_samples_leaf=50, l2_regularization=1.0,
                                       early_stopping=False, random_state=C.SEED).fit(X[itr], A[itr])
    ll = [float(-np.mean(np.log(np.clip(p[np.arange(iva.sum()), A[iva]], 1e-12, 1)))) for p in m.staged_predict_proba(X[iva])]
    nt = int(np.argmin(ll)) + 1
    pb = next(p for i, p in enumerate(m.staged_predict_proba(X[te])) if i + 1 == nt)[:, 2]
    pg = T.p2.to_numpy()[te]
    y = (A[te] == 2).astype(int)
    gi, gu = pd.factorize(mids[te])
    rng = np.random.default_rng(seed)
    bs = {"gat": [], "ball_only": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu))[gi]
        bs["gat"].append(roc_auc_score(y, pg, sample_weight=w))
        bs["ball_only"].append(roc_auc_score(y, pb, sample_weight=w))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"held_out_season": C.SEASONS[1], "n": int(te.sum()), "n_matches": int(len(gu)), "cross_rate": float(y.mean()),
            "gat": {"auc": float(roc_auc_score(y, pg)), "ci": ci(bs["gat"])},
            "ball_only": {"auc": float(roc_auc_score(y, pb)), "ci": ci(bs["ball_only"]), "trees": nt},
            "difference": {"auc": float(roc_auc_score(y, pg) - roc_auc_score(y, pb)),
                           "ci": ci(np.array(bs["gat"]) - np.array(bs["ball_only"]))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder for the JSON results")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--propensities", help="skip training and evaluate these propensities")
    a = ap.parse_args()
    out = Path(a.out)
    z = np.load(out / "choice_inputs.npz", allow_pickle=True)
    keep = z["low_dec"]
    D = {k: z[k][keep] for k in z.files}
    info = []
    if a.propensities:
        T = pd.read_parquet(a.propensities)
        T["season"] = T.season.astype(str)
        info.append({"train_season": C.SEASONS[0], "source": "saved propensities",
                     "val_matches": validation_matches(D["match_id"], D["season"].astype(str), C.SEASONS[0], C.SEED_GAT)})
    else:
        P = np.full((keep.sum(), 3), np.nan)
        for tr in C.SEASONS:
            p, i = train_season_swap(D["W"], D["M"], D["ctx"], D["A"].astype(int), D["season"].astype(str), D["match_id"],
                                     tr, seed=C.SEED_GAT, epochs=a.epochs, minutes=a.minutes, device=a.device)
            te = D["season"].astype(str) != tr
            P[te] = p[te]
            info.append(i)
        T = pd.DataFrame({k: D[k] for k in KEYS})
        T["season"] = T.season.astype(str)
        T[["p0", "p1", "p2"]] = P
        T.to_parquet(out / "propensities.parquet")
    tab = pd.read_parquet(out / "choice_table.parquet")
    tab = tab[tab.low_dec].reset_index(drop=True)
    tab = tab.merge(T, on=KEYS, how="left")
    res = {"training": [{k: v for k, v in i.items() if k != "val_matches"} for i in info],
           "cross_auc": ball_only_auc(tab, tab.choice.to_numpy(int), info[0]["val_matches"])}
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "choice_model.json").write_text(json.dumps(res, indent=1))
    r = res["cross_auc"]
    print(f"held-out {r['held_out_season']}: cross AUC graph model {r['gat']['auc']:.3f} {np.round(r['gat']['ci'], 3).tolist()} "
          f"vs ball position only {r['ball_only']['auc']:.3f} {np.round(r['ball_only']['ci'], 3).tolist()} | n {r['n']}")


if __name__ == "__main__":
    main()
