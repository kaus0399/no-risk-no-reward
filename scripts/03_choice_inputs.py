"""Inputs for the choice models: every wide decision with a valid picture at the decision moment.

Writes to --out:
  choice_inputs.npz     graph inputs (nodes, masks, 7 context values), choice, keys, low-block flag
  choice_table.parquet  hand-built decision features, context, outcomes and choices for the gradient-boosting models

Usage: python scripts/03_choice_inputs.py --out derived
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowblock.choice_data import CONTEXT, CONTEXT_GBM, add_context, mirrored, tabular_features  # noqa: E402
from lowblock.prep import prepare  # noqa: E402

KEYS = ["season", "match_id", "team_id", "possession", "k"]
KEEP = KEYS + ["opp_id", "period", "t", "choice", "choice_type", "low_dec", "rest_xg", "opp_xg20", "net_xg", "turn5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    S = pd.read_parquet(out / "steps.parquet")
    A = pd.read_parquet(out / "attacks.parquet")
    st = pd.read_parquet(out / "strength.parquet")
    nd, md = np.load(out / "nodes_dec.npy"), np.load(out / "mask_dec.npy")
    P = add_context(prepare(S, A, st), st)
    P = P[P.wide & md[P.row.to_numpy(), 22]].reset_index(drop=True)          # ball visible at the decision moment
    r = P.row.to_numpy()
    W, M = mirrored(nd[r], md[r]), md[r]
    np.savez(out / "choice_inputs.npz", W=W, M=M, A=P.choice.to_numpy(np.int8), ctx=P[CONTEXT].to_numpy(np.float32),
             low_dec=P.low_dec.to_numpy(bool), **{k: P[k].to_numpy() for k in KEYS})
    FE = tabular_features(nd[r], md[r])
    pd.concat([P[KEEP + [c for c in CONTEXT_GBM if c not in KEEP]], FE], axis=1).to_parquet(out / "choice_table.parquet")
    lo = P.low_dec.to_numpy(bool)
    print(f"wide decisions with a picture {len(P)} | block still low {int(lo.sum())} | choice shares (low) "
          f"{np.round(np.bincount(P.choice[lo], minlength=3) / lo.sum(), 4).tolist()} | by season "
          f"{P[lo].season.value_counts().sort_index().to_dict()} | features {FE.shape[1]}")


if __name__ == "__main__":
    main()
