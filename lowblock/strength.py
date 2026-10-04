"""Team strength: the team's season-average non-penalty xG difference per match over its other matches (the current
match is left out)."""
from pathlib import Path

import pandas as pd

from .io import list_matches, read_events, read_metadata
from .schema import TEAM_STATS, load_schema, source_columns, to_generic


def match_npxgd(path, schema=None):
    """(match id, [(team id, non-penalty xG for minus against)]) for one match. Uses the team-stats file when present;
    otherwise sums shot xG from the events, excluding penalties and shoot-outs (a close approximation)."""
    s = schema or load_schema()
    meta = read_metadata(path, s)
    f = Path(path) / s["files"]["team_stats"]
    if f.exists():
        t = to_generic(pd.read_csv(f, usecols=source_columns(s["team_stats"], TEAM_STATS)), s["team_stats"], TEAM_STATS)
        if len(t) == 2:
            return meta["match_id"], [(int(r.team_id), float(r.np_xg_for - r.np_xg_against)) for r in t.itertuples()]
    ev = read_events(path, s)
    sh = ev[ev.type.eq("shot") & ~ev.shot_type.eq("penalty") & (ev.period <= 4)]
    teams = (meta["home_team_id"], meta["away_team_id"])
    xg = {t: float(sh[sh.team_id == t].xg.sum()) for t in teams}
    return meta["match_id"], [(t, xg[t] - xg[o]) for t, o in (teams, teams[::-1])]


def team_strength(data_root, seasons, schema=None):
    rows = []
    for season in seasons:
        for p in list_matches(data_root, [season], schema):
            mid, vals = match_npxgd(p, schema)
            rows += [{"season": season, "match_id": mid, "team_id": t, "npxgd": v} for t, v in vals]
    d = pd.DataFrame(rows)
    g = d.groupby(["season", "team_id"])
    n = g.npxgd.transform("size")
    d["npxgd_pm"] = (g.npxgd.transform("sum") - d.npxgd) / (n - 1)
    return d.drop(columns="npxgd")
