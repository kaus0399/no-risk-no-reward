"""Figure 1b: risk (odds of losing the ball within 5 s) and reward (net xG over the rest of the attack, as a ratio) of
each option against not taking it, at every moment the block is still low; adjusted for the ball's zone, both teams'
strength, score, minute, seconds since the block set and how the possession started; match-clustered intervals.

Writes results/figure1b.json.

Usage: python scripts/06_option_ratios.py --out derived
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowblock.prep import prepare  # noqa: E402
from lowblock.ratios import option_ratios  # noqa: E402

SINGLE = {"pass_into_block": "pass into the block", "carry_into_block": "carry into the block", "switch": "switch",
          "extra_attacker": "extra attacker near the ball", "cross": "cross"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder for the JSON results")
    a = ap.parse_args()
    out = Path(a.out)
    S = prepare(pd.read_parquet(out / "steps.parquet"), pd.read_parquet(out / "attacks.parquet"),
                pd.read_parquet(out / "strength.parquet"))
    M = S[S.low_dec].reset_index(drop=True)
    pick = lambda r, t: {"risk_turnover_odds_ratio": [r["turnover_or"][t][q] for q in ("rr", "lo", "hi")],
                         "reward_net_xg_ratio": [r["net_ratio"][t][q] for q in ("rr", "lo", "hi")],
                         "xg_rate_ratio": [r["xg_rr"][t][q] for q in ("rr", "lo", "hi")],
                         "counter_xg_rate_ratio": [r["counter_xg_rr"][t][q] for q in ("rr", "lo", "hi")]}
    res = {"n_moments": int(len(M)), "options": {}}
    for col, lab in SINGLE.items():
        res["options"][lab] = pick(option_ratios(M, [col], M[col] == 0), col)
    r = option_ratios(M, ["cross_type_lofted", "cross_type_low"], M.cross == 0)
    res["options"]["lofted cross"] = pick(r, "cross_type_lofted")
    res["options"]["low cross"] = pick(r, "cross_type_low")
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "figure1b.json").write_text(json.dumps(res, indent=1))
    for lab, v in res["options"].items():
        x, y = v["risk_turnover_odds_ratio"], v["reward_net_xg_ratio"]
        print(f"{lab}: risk {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}] | reward {y[0]:.2f} [{y[1]:.2f}, {y[2]:.2f}]")


if __name__ == "__main__":
    main()
