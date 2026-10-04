"""Turn the step table into the analysis table: rest-of-attack outcomes, turnovers, counter xG, zones, context
categories, the low-block flag at the decision moment and the wide-decision choice."""
import numpy as np
import pandas as pd

from . import config as C
from .low_block import low_at_decision

TER = ["weak", "middle", "strong"]
TSS_BINS, TSS_LAB = [0, 5, 10, 20, 40, np.inf], ["0-5", "5-10", "10-20", "20-40", "40+"]
MIN_BINS, MIN_LAB = [-np.inf, 30, 60, 75, np.inf], ["0-30", "30-60", "60-75", "75+"]
ZONE_BINS, ZONE_LAB = [-np.inf, 12, 18, 25, 35, np.inf], ["<12", "12-18", "18-25", "25-35", "35+"]
OPTIONS = ["pass_into_block", "carry_into_block", "switch", "cross", "extra_attacker"]


def zone_of(depth, y):
    z = pd.cut(depth, ZONE_BINS, labels=ZONE_LAB).astype(str) + "|" + np.where(np.abs(y) <= C.BOX_HALF_WIDTH, "central", "wide")
    return z.where(~z.str.startswith("nan"), np.nan)


def strength_terciles(strength, seasons):
    st = strength[strength.season.isin(seasons)].copy()
    st["ter"] = st.groupby("season").npxgd_pm.transform(lambda v: pd.qcut(v, 3, labels=TER).astype(str))
    return st


def prepare(steps, attacks, strength, seasons=C.SEASONS):
    """Analysis table, one row per step with a known zone at the decision moment. Row order: season, match,
    possession, k."""
    S = steps.copy()
    st = strength_terciles(strength, seasons)
    tmap = st.set_index(["match_id", "team_id"]).ter
    S["att_ter"] = [tmap.get((m, t), "middle") for m, t in zip(S.match_id, S.team_id)]
    S["def_ter"] = [tmap.get((m, t), "middle") for m, t in zip(S.match_id, S.opp_id)]
    S["score"] = np.select([S.score_diff < 0, S.score_diff > 0], ["losing", "winning"], "drawing")
    S["minb"] = pd.cut(S.minute, MIN_BINS, labels=MIN_LAB, right=False).astype(str)
    S["tss"] = pd.cut(S.k, TSS_BINS, labels=TSS_LAB, right=False).astype(str)
    S["zone_t"] = zone_of(S.ball_depth, S.ball_y)
    for c in OPTIONS + ["throw_in"]:
        S[c] = S[c].astype(float)
    S["low_t"], S["low_dec"] = low_at_decision(S)
    # the decision moment is the attack's step max(k-5, 0)
    S["_key"] = S.season + "_" + S.match_id.astype(str) + "_" + S.possession.astype(str)
    S["k_start"] = np.where(S.k >= 5, S.k - 5, 0)
    look = S[["_key", "k", "zone_t", "ball_depth", "ball_y"]].rename(
        columns={"k": "k_start", "zone_t": "zone", "ball_depth": "depth_start", "ball_y": "ball_y_start"})
    S = S.merge(look, on=["_key", "k_start"], how="left")
    S = S.merge(attacks[["season", "match_id", "possession", "t_end", "next_team", "opp_xg20", "loss_dist"]],
                on=["season", "match_id", "possession"], how="left")
    S = S.sort_values(["season", "match_id", "possession", "k"]).reset_index(drop=True)
    g = S.groupby(["season", "match_id", "possession"], sort=False)
    S["rest_shot"] = g.shots_1s.transform(lambda v: v[::-1].cumsum()[::-1])
    S["rest_xg"] = g.xg_1s.transform(lambda v: v[::-1].cumsum()[::-1])
    S["turn5"] = ((S.t_end - S.t <= C.WINDOW) & (S.next_team == S.opp_id) & (S.rest_shot == 0)).astype(float)
    S["opp_xg20"] = S.opp_xg20.fillna(0.0)
    S["net_xg"] = S.rest_xg - S.opp_xg20
    S = S[S.zone.notna()].reset_index(drop=True)
    S["wide"] = (S.ball_y_start.abs() > C.BOX_HALF_WIDTH) & (S.depth_start >= C.WIDE_DEPTH[0]) & \
                (S.depth_start <= C.WIDE_DEPTH[1])
    S["ctype"] = np.select([S.cross_cutback.astype(bool), S.cross_lofted.astype(bool), S.cross.astype(bool)],
                           ["cutback", "cross_high", "cross_low"], "")
    into = S.pass_into_block.astype(bool) | S.carry_into_block.astype(bool)
    S["choice"] = np.select([S.cross.astype(bool), into], [2, 1], 0)            # 0 recycle, 1 into the block, 2 cross
    S["choice_type"] = np.select([S.ctype == "cutback", S.ctype == "cross_high", S.ctype == "cross_low",
                                  into & (S.ctype == "")], [4, 2, 3, 1], 0)    # by cross type
    S["cross_type_lofted"] = (S.cross.astype(bool) & S.cross_lofted.astype(bool)).astype(float)
    S["cross_type_low"] = (S.cross.astype(bool) & ~S.cross_lofted.astype(bool)).astype(float)
    return S
