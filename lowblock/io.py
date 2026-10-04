"""Read one match folder and align the event clock with the tracking clock.

Layout: <data_root>/<season>/<match>/
    match_metadata.json
    tracking/tracking.parquet
    events/events.csv
    events/team_stats.csv   (optional; team strength falls back to shot xG from the events)
    events/positions.csv    (optional; only for matches without player positions in the tracking file)
File, column and value names can be remapped with a schema file (lowblock/schema.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import EVENTS, METADATA, POSITIONS, TRACKING, load_schema, source_columns, to_generic

PERIOD_START_S = {1: 0.0, 2: 2700.0, 3: 5400.0, 4: 6300.0}


def list_matches(data_root, seasons=None, schema=None):
    s = schema or load_schema()
    folders = sorted(p.parent for p in Path(data_root).glob(f"*/*/{s['files']['metadata']}"))
    if seasons:
        folders = [f for f in folders if f.parent.name in seasons]
    return folders


def read_metadata(path, schema=None):
    s = schema or load_schema()
    raw = json.loads((Path(path) / s["files"]["metadata"]).read_text())
    meta = {k: raw[s["metadata"].get(k, k)] for k in METADATA}
    meta["match_id"], meta["home_team_id"], meta["away_team_id"] = (int(meta["match_id"]), int(meta["home_team_id"]),
                                                                    int(meta["away_team_id"]))
    meta["direction_by_period"] = {str(k): bool(v) for k, v in meta["direction_by_period"].items()}
    return meta


def period_clock(df):
    """game_clock in seconds from the start of the period, whether or not the feed resets each period."""
    start = df["period"].map(PERIOD_START_S)
    runs_on = df.groupby("period")["game_clock"].transform("min") >= start - 5
    return np.where(runs_on & (df["period"] > 1), df["game_clock"] - start, df["game_clock"])


def event_seconds(ts):
    """Event time in seconds from the start of the period: numbers as they are, 'HH:MM:SS.mmm' strings converted."""
    if pd.api.types.is_numeric_dtype(ts):
        return ts.astype(float)
    parts = ts.astype(str).str.split(":", expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def attacks_positive_x(meta, period, is_home):
    home_pos = bool(meta["direction_by_period"][str(period)])
    return home_pos if is_home else not home_pos


def events_to_tracking(x_ev, y_ev, length, width, attacks_pos):
    """Event coordinates (120 x 80 grid, the acting team attacking +x) -> tracking metres (origin at the centre spot)."""
    x = np.asarray(x_ev, float) / 120.0 * length - length / 2
    y = width / 2 - np.asarray(y_ev, float) / 80.0 * width
    return (x, y) if attacks_pos else (-x, -y)


def clock_offsets(meta, tracking, events, search=(-4.0, 2.0), step=0.04):
    """Per-period offset (tracking time = event time + offset): the shift that best lines up pass start locations with
    the tracked ball. Returns ({period: offset}, {period: median distance in metres})."""
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team_id"]
    ball = tracking[tracking.home_away_ball == "ball"][["period", "game_clock", "live", "x", "y"]].copy()
    ball["t"] = period_clock(ball)
    ball = ball[ball.live].sort_values("t")
    passes = events[(events.type == "pass") & events.location_x.notna()]
    off, dist = {}, {}
    for period, p in passes.groupby("period"):
        bp = ball[ball.period == period]
        if len(bp) < 100:
            continue
        is_home = (p.team_id == home).to_numpy()
        ex, ey = np.empty(len(p)), np.empty(len(p))
        for flag in (True, False):
            m = is_home == flag
            ex[m], ey[m] = events_to_tracking(p.location_x[m], p.location_y[m], length, width,
                                              attacks_positive_x(meta, int(period), flag))
        t = bp.t.to_numpy()
        bx, by = bp.x.to_numpy(), bp.y.to_numpy()
        best = None
        for o in np.arange(search[0], search[1] + 1e-9, step):
            q = p.t.to_numpy() + o
            idx = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
            idx = np.where(np.abs(t[idx - 1] - q) < np.abs(t[idx] - q), idx - 1, idx)
            d = np.median(np.hypot(bx[idx] - ex, by[idx] - ey))
            if best is None or d < best[1]:
                best = (o, d)
        off[int(period)], dist[int(period)] = round(best[0], 2), round(best[1], 2)
    return off, dist


def read_events(path, schema=None):
    s = schema or load_schema()
    f = Path(path) / s["files"]["events"]
    hdr = pd.read_csv(f, nrows=0).columns
    raw = pd.read_csv(f, low_memory=False, usecols=[c for c in source_columns(s["events"], EVENTS) if c in hdr])
    ev = to_generic(raw, s["events"], EVENTS)
    gx, gy = s["events"].get("grid", [120, 80])
    if (gx, gy) != (120, 80):                                 # work on a 120 x 80 grid internally
        for c, g in (("location_x", gx), ("end_location_x", gx), ("location_y", gy), ("end_location_y", gy)):
            ev[c] = ev[c] * (120.0 if c.endswith("x") else 80.0) / g
    return ev


def load_match(path, schema=None):
    """Metadata, tracking (all frames) and events with an aligned clock column `tt` (seconds in the period)."""
    s = schema or load_schema()
    path = Path(path)
    meta = read_metadata(path, s)
    raw = pd.read_parquet(path / s["files"]["tracking"], columns=source_columns(s["tracking"], TRACKING))
    tr = to_generic(raw, s["tracking"], TRACKING)
    del raw
    tr["frame_idx"] = tr.frame_idx.astype("int64")
    tr["period"] = tr.period.astype(int)
    ev = read_events(path, s)
    ev["t"] = event_seconds(ev.timestamp)
    ev = ev.sort_values(["period", "t", "event_id"]).reset_index(drop=True)
    off, dist = clock_offsets(meta, tr, ev)
    ev["tt"] = ev.t + ev.period.map(off).fillna(0.0)
    ev["dur"] = ev.duration.fillna(0.0).clip(0, 10)
    positions = None
    f = path / s["files"]["positions"]
    if f.exists():
        praw = pd.read_csv(f, usecols=source_columns(s["positions"], POSITIONS))
        positions = to_generic(praw, s["positions"], POSITIONS)
    return {"meta": meta, "tracking": tr, "events": ev, "offsets": off, "sync_dist": dist, "positions": positions}
