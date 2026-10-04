"""How much possession is spent against a set low block that is still low: share of in-possession live time, overall
and for the strongest and weakest third of teams (team-season bootstrap), and the trend across seasons.

Inputs in --out: steps.parquet, attacks.parquet, inposs.parquet, strength.parquet; for the trend also
steps_light.parquet and inposs_light.parquet (01_build_steps.py --light over the trend seasons).
Writes results/low_block_share.json.

Usage: python scripts/07_low_block_share.py --out derived
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowblock import config as C  # noqa: E402
from lowblock.prep import TER, strength_terciles  # noqa: E402


def team_matches(steps, inposs, strength, seasons, depth_col, by_inposs=False):
    """Low-block seconds and in-possession seconds per team-match (rows ordered by the strength table, or by the
    possession table when by_inposs)."""
    st = strength_terciles(strength, seasons)[["season", "match_id", "team_id", "ter"]]
    S = steps[steps.season.isin(seasons)].assign(low_t=lambda d: d[depth_col] <= C.DEEP3_MAX)
    g = S.groupby(["season", "match_id", "team_id"]).low_t.sum().rename("low_t_s").reset_index()
    ip = inposs[inposs.season.isin(seasons)]
    k = ["season", "match_id", "team_id"]
    tm = ip.merge(st, on=k, how="left") if by_inposs else st.merge(ip, on=k, how="inner")
    tm = tm.merge(g, on=["season", "match_id", "team_id"], how="left").fillna({"low_t_s": 0})
    tm["ts"] = tm.season + "_" + tm.team_id.astype(str)
    return tm


def shares(d):
    out = {"all": d.low_t_s.sum() / d.inposs_s.sum()}
    out.update({t: d[d.ter == t].low_t_s.sum() / d[d.ter == t].inposs_s.sum() for t in TER})
    return out


def team_bootstrap(tm, nb, rng):
    keys = tm.ts.unique()
    grp = {k: v for k, v in tm.groupby("ts")}
    return [shares(pd.concat([grp[k] for k in rng.choice(keys, len(keys), replace=True)], ignore_index=True))
            for _ in range(nb)]


def summary(pt, B):
    res = {k: [100 * pt[k], 100 * float(np.percentile([b[k] for b in B], 2.5)),
               100 * float(np.percentile([b[k] for b in B], 97.5))] for k in pt}
    r = [b["strong"] / b["weak"] for b in B]
    res["strong_vs_weak"] = [pt["strong"] / pt["weak"], float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder for the JSON results")
    a = ap.parse_args()
    out = Path(a.out)
    st = pd.read_parquet(out / "strength.parquet")
    tm = team_matches(pd.read_parquet(out / "steps.parquet"), pd.read_parquet(out / "inposs.parquet"), st, C.SEASONS,
                      "deep3_nodes_t")
    res = {"pooled_" + "_".join(C.SEASONS): summary(shares(tm), team_bootstrap(tm, C.N_BOOT, np.random.default_rng(C.SEED_SHARE)))}
    p = res["pooled_" + "_".join(C.SEASONS)]
    print(f"share of in-possession time against a set low block that is still low: {p['all'][0]:.1f}% "
          f"[{p['all'][1]:.1f}, {p['all'][2]:.1f}] | strongest third {p['strong'][0]:.1f}% | weakest third {p['weak'][0]:.1f}%")
    if (out / "steps_light.parquet").exists():
        L, IL = pd.read_parquet(out / "steps_light.parquet"), pd.read_parquet(out / "inposs_light.parquet")
        seasons = sorted(IL.season.unique())
        rng = np.random.default_rng(C.SEED_SHARE_SEASON)
        trend, B = {}, {}
        all_matches = team_matches(L, IL, st, seasons, "deep3_t", by_inposs=True)
        for s in seasons:
            d = all_matches[all_matches.season == s]
            B[s] = team_bootstrap(d, C.N_BOOT_SEASON, rng)
            trend[s] = summary(shares(d), B[s])
        first, last = seasons[0], seasons[-1]
        diff = [100 * (bl["all"] - bf["all"]) for bf, bl in zip(B[first], B[last])]
        x = np.arange(len(seasons))
        slopes = [100 * np.polyfit(x, [B[s][i]["all"] for s in seasons], 1)[0] for i in range(C.N_BOOT_SEASON)]
        res["by_season"] = trend
        res["change_first_to_last_pct_points"] = [trend[last]["all"][0] - trend[first]["all"][0],
                                                  float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        res["slope_pct_points_per_season"] = [float(np.polyfit(x, [trend[s]["all"][0] for s in seasons], 1)[0]),
                                              float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
        c = res["change_first_to_last_pct_points"]
        print("by season:", {s: round(v["all"][0], 1) for s, v in trend.items()},
              f"| change {first} -> {last}: {c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}] points")
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "low_block_share.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
