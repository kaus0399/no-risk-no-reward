"""Input schema: the generic file, column and value names the pipeline reads, and an optional JSON file that maps
them to the column names used in another dataset.

Folder per match: DATA/<season>/<match>/ with match_metadata.json, tracking/tracking.parquet, events/events.csv,
and optionally events/team_stats.csv and events/positions.csv. Season folders are named like 20242025; the seasons
used are set in lowblock/config.py. Team and player ids are integers shared by the tracking and event files.

Metadata: match_id, home_team_id, away_team_id, pitch_length and pitch_width (metres), direction_by_period
  ({"1": true, "2": false} means the home team attacks towards +x in period 1).

Tracking, one row per player or ball per frame at 25 frames per second:
  frame_idx (int, continuous across periods), period (1, 2), game_clock (seconds; may restart each period or run
  on), live (bool, ball in play), home_away_ball ("home", "away", "ball"), is_team_in_possession (bool),
  team_id and player_id (empty for the ball), player_position ("goalkeeper" marks keepers),
  x and y (metres from the centre spot; x along the length, y positive to the left of a team attacking +x).

Events, one row per event, coordinates on a 120 x 80 grid seen by the acting team (x from its own goal line to the
opponent's, y from its left touchline to its right; "grid" in the mapping file rescales other grids):
  event_id, period, timestamp ("HH:MM:SS.mmm" or seconds from the start of the period), possession (a number that
  increases through the match), possession_team_id, team_id,
  type ("pass", "carry", "shot", "ball_recovery", "interception", "duel", "own_goal"; other types are ignored),
  play_pattern ("from_corner", "from_free_kick", "from_throw_in", "other"), location_x, location_y,
  end_location_x, end_location_y (end of a pass or carry), duration (seconds),
  pass_outcome ("complete", "incomplete"), pass_type ("corner", "free_kick", "other"),
  pass_height ("high", "low", "ground"), cross and cut_back (bool), shot_type ("free_kick", "penalty", "other"),
  shot_outcome ("goal", "other"), xg, recovery_failed (bool), interception_outcome ("won", "lost"),
  duel_type ("tackle", "other"), duel_outcome ("won", "lost"), card ("yellow", "red").

Optional: team_stats (team_id, np_xg_for, np_xg_against per match; otherwise team strength uses shot xG from the
events) and positions (player_id, position, seconds; only needed if the tracking file has no player positions).

Mapping file format (every section optional; anything not listed keeps its generic name):
{
  "files":     {"metadata": "...", "tracking": "...", "events": "...", "team_stats": "...", "positions": "..."},
  "metadata":  {"match_id": "<key in the metadata file>", ...},
  "tracking":  {"columns": {"team_id": "<column>", ...}, "values": {"player_position": {"<value>": "goalkeeper"}}},
  "events":    {"columns": {"end_location_x": ["<column a>", "<column b>"], ...},
                "values": {"type": {"<value>": "pass", ...}, "pass_outcome": {"<value>": "complete",
                                                                               "__other__": "incomplete"}},
                "grid": [120, 80]},
  "team_stats": {"columns": {...}},
  "positions":  {"columns": {...}, "values": {...}}
}
A list of source columns is read as the first non-missing value per row. In value maps, "__null__" maps missing
values and "__other__" maps every value not listed; values without a mapping are kept as they are.
"""
import copy
import json

import numpy as np
import pandas as pd

FILES = {"metadata": "match_metadata.json", "tracking": "tracking/tracking.parquet", "events": "events/events.csv",
         "team_stats": "events/team_stats.csv", "positions": "events/positions.csv"}
METADATA = ["match_id", "home_team_id", "away_team_id", "pitch_length", "pitch_width", "direction_by_period"]
TRACKING = ["frame_idx", "period", "game_clock", "live", "home_away_ball", "is_team_in_possession", "team_id",
            "player_id", "player_position", "x", "y"]
EVENTS = ["event_id", "period", "timestamp", "type", "possession", "possession_team_id", "team_id", "play_pattern",
          "location_x", "location_y", "end_location_x", "end_location_y", "duration", "pass_outcome", "pass_type",
          "pass_height", "cross", "cut_back", "shot_type", "shot_outcome", "xg", "recovery_failed",
          "interception_outcome", "duel_type", "duel_outcome", "card"]
TEAM_STATS = ["team_id", "np_xg_for", "np_xg_against"]
POSITIONS = ["player_id", "position", "seconds"]
BOOLEAN = {"live", "is_team_in_possession", "cross", "cut_back", "recovery_failed"}
GRID = (120.0, 80.0)                                    # event coordinates: length x width, team attacking +x

DEFAULT = {"files": FILES, "metadata": {k: k for k in METADATA}, "tracking": {"columns": {}, "values": {}},
           "events": {"columns": {}, "values": {}, "grid": list(GRID)}, "team_stats": {"columns": {}},
           "positions": {"columns": {}, "values": {}}}


def load_schema(path=None):
    s = copy.deepcopy(DEFAULT)
    if path:
        user = json.loads(open(path).read())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(s.get(k), dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and isinstance(s[k].get(kk), dict):
                        s[k][kk].update(vv)
                    else:
                        s[k][kk] = vv
            else:
                s[k] = v
    return s


def source_columns(section, generic):
    """Source column names needed to build the generic columns of one table."""
    cols = []
    for g in generic:
        src = section["columns"].get(g, g)
        cols += src if isinstance(src, list) else [src]
    return cols


def _map_values(s, m):
    out = s.astype(object).copy()
    null = s.isna()
    known = s.isin(list(m.keys())) & ~null
    out[known] = s[known].map(m)
    if "__other__" in m:
        out[~known & ~null] = m["__other__"]
    if "__null__" in m:
        out[null] = m["__null__"]
    return out


def to_generic(df, section, generic):
    """Rename (or coalesce) source columns to the generic names and map values; missing columns become NaN."""
    out = {}
    for g in generic:
        src = section["columns"].get(g, g)
        srcs = [c for c in (src if isinstance(src, list) else [src]) if c in df]
        if not srcs:
            col = pd.Series(np.nan, index=df.index)
        elif len(srcs) == 1:
            col = df[srcs[0]]
        else:
            col = df[srcs[0]]
            for c in srcs[1:]:
                col = col.where(col.notna(), df[c])
        if g in section.get("values", {}):
            col = _map_values(col, section["values"][g])
        if g in BOOLEAN:
            col = col if col.dtype == bool else col.astype(str).str.lower().isin(["true", "1", "1.0"])
        elif col.dtype == object:
            num = pd.to_numeric(col, errors="coerce")
            if num.notna().sum() == col.notna().sum():
                col = num
        out[g] = col
    return pd.DataFrame(out, index=df.index)
