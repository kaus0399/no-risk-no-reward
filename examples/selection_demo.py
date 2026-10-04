"""Why options must be compared at equal likelihood: a simulation with known answers.

Decisions are simulated with three options (recycle, pass or carry into the block, cross). Teams are more likely to
cross or play into the block when the situation already favours a chance, so a raw comparison of outcomes credits the
option with value that belongs to the situation. The doubly robust estimator in lowblock/aipw.py, with choice and
outcome models fitted by cross-validation, recovers the true prices.

No match data is needed. Run from the repository root:
    python examples/selection_demo.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowblock.aipw import folds_by_match, price  # noqa: E402
from lowblock.outcome_models import crossfit_choice, crossfit_outcome  # noqa: E402

NAMES = ["recycle", "pass or carry into the block", "cross"]
TRUE_XG = np.array([0.0, 0.0030, 0.0085])       # xG added per decision, against recycling


def simulate(n_matches=400, per_match=250, seed=1):
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    match = np.repeat(np.arange(n_matches), per_match)
    depth = rng.uniform(12, 35, n)                # metres from goal
    width = rng.uniform(20, 34, n)                # metres from the centre line
    space = rng.normal(0, 1, n)                   # how open the situation is
    X = np.column_stack([depth, width, space])
    # the situation that makes an option attractive also makes a chance more likely
    on = 0.9 * space - 0.05 * (depth - 23)
    logits = np.column_stack([np.zeros(n), 0.2 + 0.6 * on, -1.6 + 1.0 * on + 0.08 * (width - 27)])
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    A = (rng.uniform(size=(n, 1)) > p.cumsum(1)).sum(1)
    base_xg = 0.02 * np.exp(0.5 * on)
    rest_xg = rng.gamma(0.5, 2 * (base_xg + TRUE_XG[A]))
    return X, A, match, {"rest_xg": rest_xg}


def main(n_runs=5):
    raw, est, cover = {1: [], 2: []}, {1: [], 2: []}, {1: 0, 2: 0}
    for seed in range(1, n_runs + 1):
        X, A, match, Y = simulate(seed=seed)
        fold = folds_by_match(match)
        e = crossfit_choice(X, A, 3, fold)
        mu = {"rest_xg": crossfit_outcome(X, A, Y["rest_xg"], 3, fold, "reg")}
        for a in (1, 2):
            raw[a].append(100 * (Y["rest_xg"][A == a].mean() - Y["rest_xg"][A == 0].mean()))
            pt, lo, hi = price(e, mu, A, {"rest_xg": Y["rest_xg"]}, a, match, n_boot=300)["rest_xg"]
            est[a].append(pt)
            cover[a] += int(lo <= 100 * TRUE_XG[a] <= hi)
    print(f"{n_runs} simulations of {len(A)} decisions each; xG per 100 decisions, against recycling\n")
    print(f"{'option':32s} {'true':>6s} {'raw':>6s} {'doubly robust':>14s} {'intervals containing the truth':>32s}")
    for a in (1, 2):
        print(f"{NAMES[a]:32s} {100 * TRUE_XG[a]:6.2f} {np.mean(raw[a]):6.2f} {np.mean(est[a]):14.2f} {cover[a]:>26d} of {n_runs}")
    ok = all(np.mean(raw[a]) > 100 * TRUE_XG[a] and abs(np.mean(est[a]) / (100 * TRUE_XG[a]) - 1) < 0.1 for a in (1, 2))
    if ok:
        print("\nThe raw comparison overstates both options; the doubly robust prices are within 10% of the true values.")


if __name__ == "__main__":
    main()
