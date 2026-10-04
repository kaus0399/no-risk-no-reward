"""Find attacks against a set low block and build the 1-s decision steps for every match.

Writes to --out:
  steps.parquet, attacks.parquet, inposs.parquet, nodes_dec.npy, mask_dec.npy   (default)
  steps_light.parquet, attacks_light.parquet, inposs_light.parquet               (--light: no options or nodes,
                                                                                  for the share-of-possession trend)
One checkpoint per match in --out/parts[_light]/.

Usage:
  python scripts/01_build_steps.py --data-root DATA --out derived [--schema schema.json] [--seasons 20242025,20252026]
  python scripts/01_build_steps.py --data-root DATA --out derived --light --seasons 20212022,...,20252026
"""
import os

# one thread per process; four processes run in parallel
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowblock import config as C  # noqa: E402
from lowblock.io import list_matches, read_metadata  # noqa: E402
from lowblock.schema import load_schema  # noqa: E402
from lowblock.steps import build_match  # noqa: E402


def one(job):
    path, season, parts, light, schema_path = job
    schema = load_schema(schema_path)
    mid = read_metadata(path, schema)["match_id"]
    ck = parts / f"{season}_{mid}.pkl"
    if ck.exists():
        return ck
    try:
        ck.write_bytes(pickle.dumps(build_match(path, season, light, schema), protocol=5))
        return ck
    except Exception as ex:                                  # report and continue with the other matches
        return f"{season}_{mid}: {type(ex).__name__}: {str(ex)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", help="JSON file mapping the generic input names to your dataset's column names")
    ap.add_argument("--seasons", default=",".join(C.SEASONS))
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N matches per season (for a quick check)")
    a = ap.parse_args()
    root, out = C.data_root(a.data_root), Path(a.out)
    sfx = "_light" if a.light else ""
    parts = out / f"parts{sfx}"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in a.seasons.split(","):
        ps = list_matches(root, [s], load_schema(a.schema))
        jobs += [(p, s, parts, a.light, a.schema) for p in (ps[: a.limit] if a.limit else ps)]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(min(a.workers, 4)) as ex:      # at most four: each worker holds a full match in memory
        for i, r in enumerate(ex.map(one, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} matches, {time.time() - t0:.0f} s", flush=True)
    errs = [r for r in res if isinstance(r, str)]
    A, S, I, ND, MD = [], [], [], [], []
    for ck in [r for r in res if not isinstance(r, str)]:
        d = pickle.loads(ck.read_bytes())
        if len(d["attacks"]):
            A.append(d["attacks"])
        if len(d["steps"]):
            S.append(d["steps"])
            if not a.light:
                ND.append(d["nodes_dec"])
                MD.append(d["mask_dec"])
        I += [(d["season"], d["mid"], t, v) for t, v in d["inposs"].items()]
    A, S = pd.concat(A, ignore_index=True), pd.concat(S, ignore_index=True)
    S["row"] = np.arange(len(S))
    A.to_parquet(out / f"attacks{sfx}.parquet")
    S.to_parquet(out / f"steps{sfx}.parquet")
    pd.DataFrame(I, columns=["season", "match_id", "team_id", "inposs_s"]).to_parquet(out / f"inposs{sfx}.parquet")
    if not a.light:
        np.save(out / "nodes_dec.npy", np.concatenate(ND))
        np.save(out / "mask_dec.npy", np.concatenate(MD))
    print(f"matches {len(jobs)} | ok {len(jobs) - len(errs)} | failed {len(errs)} | attacks {len(A)} "
          f"({len(A) / max(len(jobs) - len(errs), 1):.1f} per match) | steps {len(S)} | median attack {A.dur.median():.1f} s | "
          f"{time.time() - t0:.0f} s", flush=True)
    if errs:
        print("failed:", errs[:3])


if __name__ == "__main__":
    main()
