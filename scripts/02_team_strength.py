"""Team strength per match: the team's season-average non-penalty xG difference over its other matches.

Usage: python scripts/02_team_strength.py --data-root DATA --out derived [--schema schema.json] [--seasons ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowblock import config as C  # noqa: E402
from lowblock.schema import load_schema  # noqa: E402
from lowblock.strength import team_strength  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", help="JSON file mapping the generic input names to your dataset's column names")
    ap.add_argument("--seasons", default=",".join(C.TREND_SEASONS))
    a = ap.parse_args()
    st = team_strength(C.data_root(a.data_root), a.seasons.split(","), load_schema(a.schema))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    st.to_parquet(Path(a.out) / "strength.parquet")
    g = st.groupby("season")
    print(f"team-matches {len(st)} | per season {g.size().to_dict()} | sd of strength by season "
          f"{g.npxgd_pm.std().round(3).to_dict()}")


if __name__ == "__main__":
    main()
