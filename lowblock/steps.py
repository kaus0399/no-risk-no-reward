"""Attacks against a set low block, 1-s decision steps, the options taken in each step's window, and outcomes.

One attack = one event-data possession from the moment the block sets until the ball is lost, goes dead for 1 s or
more, or the period ends. Each second of the attack is a step at time t. The option for step t is what the attacking
team did in [t-5, t]; the decision moment is t-5.

Per match this returns:
  attacks   one row per attack
  steps     one row per step: options, outcomes, ball position, block depth, context
  nodes_t   [n_steps, 23, 4] players and ball at t (depth, lateral, vx, vy), for the low-block flag
  nodes_dec [n_steps, 23, 4] the same at the decision moment t-5, for the choice models; masks alongside
  inposs    in-possession live seconds per team
"""
import numpy as np
import pandas as pd

from . import config as C
from .io import load_match, period_clock
from .low_block import first_set, two_lines

def build_match(path, season, light=False, schema=None):
    m = load_match(path, schema)
    meta, tr, ev = m["meta"], m["tracking"], m["events"]
    mid = meta["match_id"]
    half = float(meta["pitch_length"]) / 2
    Lp = float(meta["pitch_length"])
    home, away = meta["home_team_id"], meta["away_team_id"]
    dirp = {int(p): bool(v) for p, v in meta["direction_by_period"].items()}

    # ---- 5 Hz frames: ball, players, possession flag
    tr = tr[tr.frame_idx % C.STEP == 0]
    ball = tr[tr.home_away_ball == "ball"].sort_values("frame_idx").drop_duplicates("frame_idx")
    fr = ball.frame_idx.to_numpy()
    nF = len(fr)
    per = ball.period.to_numpy()
    live = ball.live.to_numpy().astype(bool)
    tclk = np.asarray(period_clock(ball), float)
    bxy = ball[["x", "y"]].to_numpy(float)
    pl = tr[(tr.home_away_ball != "ball") & tr.player_id.notna() & tr.team_id.notna()].copy()
    pl["pid"] = pl.player_id.astype(float).astype("int64")
    pl["tid"] = pl.team_id.astype(float).astype("int64")
    pids = np.sort(pl.pid.unique())
    nS = len(pids)
    mode = lambda c: pl.groupby(["pid", c]).size().reset_index(name="n").sort_values(
        ["pid", "n", c], ascending=[True, False, True]).drop_duplicates("pid").set_index("pid")[c].reindex(pids).to_numpy()
    team_of = mode("tid")
    gk_of = (pl.player_position == "goalkeeper").groupby(pl.pid).mean().reindex(pids).to_numpy() > 0.5
    lab_of = mode("player_position")
    no_lab = pd.isna(lab_of)
    if no_lab.any() and m["positions"] is not None:          # no position in the tracking file: most-played position
        pos = m["positions"].groupby(["player_id", "position"]).seconds.sum() \
            .reset_index().sort_values("seconds").drop_duplicates("player_id", keep="last").set_index("player_id").position
        lab_of = np.array([pos.get(q, "unknown") if nl else lb for q, lb, nl in zip(pids, lab_of, no_lab)], dtype=object)
        gk_of = np.where(no_lab, lab_of == "goalkeeper", gk_of)
    fi = np.searchsorted(fr, pl.frame_idx.to_numpy())
    ok = (fi < nF) & (fr[np.clip(fi, 0, nF - 1)] == pl.frame_idx.to_numpy())
    si = np.searchsorted(pids, pl.pid.to_numpy())
    P = np.full((nF, nS, 2), np.nan)
    P[fi[ok], si[ok]] = pl[["x", "y"]].to_numpy(float)[ok]
    poss = np.zeros(nF, np.int64)
    itp = pl.is_team_in_possession.to_numpy().astype(bool) & ok
    poss[fi[itp]] = pl.tid.to_numpy()[itp]
    del tr, pl
    pid_slot = {int(q): i for i, q in enumerate(pids)}
    present = ~np.isnan(P[:, :, 0])
    eleven = (present[:, team_of == home].sum(1) == 11) & (present[:, team_of == away].sum(1) == 11)
    seg = np.cumsum(np.r_[True, (np.diff(fr) > C.FPS) | (per[1:] != per[:-1])])

    # ---- event-data context per frame: possession team and number, set-piece windows, red cards
    evposs = np.zeros(nF, np.int64)
    evno = np.full(nF, -1, np.int64)
    sp_mask = np.zeros(nF, bool)
    red_mask = np.zeros(nF, bool)
    sp = ev[(ev.type.eq("pass") & ev.pass_type.isin(["corner", "free_kick"])) | (ev.type.eq("shot") & ev.shot_type.eq("free_kick"))]
    reds = ev[ev.card.eq("red")]
    goals = ev[(ev.type.eq("shot") & ev.shot_outcome.eq("goal")) | ev.type.eq("own_goal")]
    first_red = (int(reds.period.iloc[0]), float(reds.tt.iloc[0])) if len(reds) else None
    pidx = {}
    for p in np.unique(per):
        mm = np.where(per == p)[0]
        pidx[int(p)] = mm
        e = ev[ev.period == p]
        idx = np.searchsorted(e.tt.to_numpy(), tclk[mm], side="right") - 1
        evposs[mm] = np.where(idx >= 0, e.possession_team_id.to_numpy()[np.clip(idx, 0, None)], 0)
        evno[mm] = np.where(idx >= 0, e.possession.to_numpy()[np.clip(idx, 0, None)], -1)
        for tsp in sp[sp.period == p].tt.to_numpy():
            sp_mask[mm] |= (tclk[mm] >= tsp - 0.2) & (tclk[mm] <= tsp + C.SET_PIECE_WINDOW)
        if first_red is not None:
            red_mask[mm] = (p > first_red[0]) | ((p == first_red[0]) & (tclk[mm] >= first_red[1]))

    def frame_at(p, t):
        mm = pidx.get(int(p))
        if mm is None or not len(mm):
            return -1
        j = int(np.clip(np.searchsorted(tclk[mm], t), 0, len(mm) - 1))
        if j > 0 and abs(tclk[mm][j - 1] - t) < abs(tclk[mm][j] - t):
            j -= 1
        return int(mm[j])

    def score_diff(team, p, t):
        g = goals[(goals.period < p) | ((goals.period == p) & (goals.tt < t))]
        return int((g.team_id == team).sum() - (g.team_id != team).sum())

    goal_pt = np.array([half, 0.0])
    sgn_of = lambda A, p: 1.0 if (dirp[int(p)] if A == home else not dirp[int(p)]) else -1.0
    out_slots = {t: np.where((team_of == t) & ~gk_of)[0] for t in (home, away)}
    all_slots = {t: np.where(team_of == t)[0] for t in (home, away)}

    def goalside(Pk, bk, dset):
        dd = np.linalg.norm(Pk[dset] - goal_pt, axis=1)
        return dset[(dd < np.linalg.norm(bk - goal_pt)) & ~np.isnan(Pk[dset, 0])]

    def lines_at(Pk, dset):
        q = dset[~np.isnan(Pk[dset, 0])]
        if len(q) < 2 * C.MIN_LINE:
            return None
        ln = two_lines(half - Pk[q, 0])
        return None if ln is None else (ln[0], ln[1], q[ln[2]])

    state_cache = {}

    def state(A, k):
        """(two lines found, gap, back depth, midfield depth, attackers minus defenders within 10 m) at frame k."""
        key = (A, k)
        if key in state_cache:
            return state_cache[key]
        B = away if A == home else home
        sg = sgn_of(A, per[k])
        Pk, bk = P[k] * sg, bxy[k] * sg
        out = (False, np.nan, np.nan, np.nan, np.nan)
        if not np.isnan(bk).any():
            ln = lines_at(Pk, goalside(Pk, bk, out_slots[B]))
            da = np.linalg.norm(Pk[all_slots[A]] - bk, axis=1)
            db = np.linalg.norm(Pk[all_slots[B]] - bk, axis=1)
            ovl = int(np.nansum(da <= C.OVERLOAD_R) - np.nansum(db <= C.OVERLOAD_R))
            out = (True, ln[1] - ln[0], ln[0], ln[1], ovl) if ln is not None else (False, np.nan, np.nan, np.nan, ovl)
        state_cache[key] = out
        return out

    # ---- players and ball as 23 nodes (attackers GK first, defenders GK first, ball) in the attacking team's frame
    def node_slots(T, kf):
        B = away if T == home else home
        pres = ~np.isnan(P[kf, :, 0])
        out = []
        for team in (T, B):
            s_ = np.where((team_of == team) & pres)[0]
            s_ = np.r_[s_[gk_of[s_]][:1], s_[~gk_of[s_]][:10]]
            if not gk_of[s_[:1]].any():
                s_ = np.r_[-1, s_[:10]]
            out.append(np.r_[s_, np.full(11 - len(s_), -1)].astype(int))
        return np.r_[out[0], out[1]]

    def node_frame(T, p, kf, back):
        """Nodes at the frame `back` x 0.2 s before frame kf (slots and live segment fixed at kf): positions in metres
        from the defended goal line (depth) and from the centre line (lateral), velocities in m/s. Lateral values are
        mirrored when the ball's first valid position in the 5 s before kf is on the negative side; the choice-model
        inputs are mirrored again by the ball at the decision moment (choice_data.mirrored)."""
        sl = node_slots(T, kf)
        sg = sgn_of(T, p)
        wb = fr[kf] - C.STEP * np.arange(25, -1, -1)            # the 5 s history of frame kf, for the first mirror
        ib = np.searchsorted(fr, wb)
        ibc = np.clip(ib, 0, nF - 1)
        okb = (ib < nF) & (fr[ibc] == wb) & (seg[ibc] == seg[kf]) & ~np.isnan(bxy[ibc, 0])
        fv = np.where(okb[1:])[0]
        mir = bool(bxy[ibc[fv[0] + 1], 1] * sg < 0) if len(fv) else False
        want = fr[kf] - C.STEP * np.array([back + 1, back])
        idx = np.searchsorted(fr, want)
        idc = np.clip(idx, 0, nF - 1)
        okf = (idx < nF) & (fr[idc] == want) & (seg[idc] == seg[kf])
        Q = np.full((2, 23, 2), np.nan)
        pos = np.where(sl >= 0, sl, 0)
        Q[:, :22] = P[idc][:, pos]
        Q[:, :22][:, sl < 0] = np.nan
        Q[:, 22] = bxy[idc]
        Q[~okf] = np.nan
        Q = Q * sg
        G = np.empty_like(Q)
        G[..., 0] = half - Q[..., 0]
        G[..., 1] = -Q[..., 1] if mir else Q[..., 1]
        valid = ~np.isnan(G[..., 0])
        V = (G[1] - G[0]) * 5.0
        vv = valid[1] & valid[0]
        V[~vv] = 0.0
        V[:22] = np.clip(V[:22], -C.VMAX_PLAYER, C.VMAX_PLAYER)
        V[22] = np.clip(V[22], -C.VMAX_BALL, C.VMAX_BALL)
        X = np.concatenate([np.nan_to_num(G[1], nan=0.0), V], -1).astype(np.float16)
        return X, valid[1]

    # ---- block condition per attacking team
    cond, deep3_of = {}, {}
    for A in (home, away):
        B = away if A == home else home
        sg = np.array([sgn_of(A, p) for p in per])
        Pn = P * sg[:, None, None]
        bn = bxy * sg[:, None]
        dfo = out_slots[B]
        gD = half - Pn[:, dfo, 0]
        gs = np.sort(np.where(np.isnan(gD), np.inf, gD), 1)
        deep3 = gs[:, :3].mean(1)
        dball = np.linalg.norm(bn - goal_pt, axis=1)
        behind = (np.linalg.norm(Pn[:, dfo] - goal_pt, axis=-1) < dball[:, None]).sum(1)
        hard = live & eleven & (evposs == A) & (poss == A) & ~sp_mask & ~red_mask
        deep3_of[A] = deep3
        cond[A] = (deep3 <= C.DEEP3_MAX) & (behind >= C.N_GOALSIDE) & hard
    del Pn
    dead_start = np.zeros(nF, bool)
    d = np.diff(np.r_[0, (~live).astype(np.int8), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s >= C.DEAD_FRAMES:
            dead_start[s] = True
    gap_after = np.zeros(nF, bool)                            # dead-ball time cut from the feed: a jump of more than 1 s
    gap_after[:-1] = (np.diff(fr) > C.FPS) & (per[1:] == per[:-1])

    # ---- possessions -> attacks -> steps
    evA = ev[ev.possession_team_id.isin([home, away])]
    ATT, STEPS, NT, ND, MT, MD = [], [], [], [], [], []
    for n, g in evA.groupby("possession"):
        A = int(g.possession_team_id.iloc[0])
        if A not in (home, away):
            continue
        B = away if A == home else home
        p = int(g.period.iloc[0])
        ix = np.where(evno == n)[0]
        ix = ix[per[ix] == p]
        if len(ix) < C.SET_FRAMES:
            continue
        pp = str(g.play_pattern.iloc[0])
        gA = g[g.team_id == A]
        spe = g[(g.type.eq("pass") & g.pass_type.isin(["corner", "free_kick"])) | (g.type.eq("shot") & g.shot_type.eq("free_kick"))]
        r_ = spe.head(1) if len(spe) else g[g.location_x.notna()].head(1)
        restart_x = np.nan
        if len(r_):
            restart_x = float(r_.location_x.iloc[0]) if int(r_.team_id.iloc[0]) == A else 120.0 - float(r_.location_x.iloc[0])
        if pp in ("from_corner", "from_free_kick") and restart_x >= 80:
            continue
        nxt = ev[(ev.period == p) & (ev.possession > n)]
        t_next = float(nxt.tt.iloc[0]) if len(nxt) else np.inf
        t_pend = float(tclk[pidx[p][-1]])
        c = cond[A][ix]
        ks = -1
        brk_ = np.where(np.diff(fr[ix]) > C.FPS)[0]            # the block must set within one live segment
        for a0, a1 in zip(np.r_[0, brk_ + 1], np.r_[brk_ + 1, len(ix)]):
            k_ = first_set(c[a0:a1])
            if k_ >= 0:
                ks = int(a0 + k_)
                break
        if ks < 0:
            continue
        kset = int(ix[ks])
        t_set = float(tclk[kset])
        cand = []
        dd_ = np.where(dead_start[kset:ix[-1] + 1])[0]
        if len(dd_):
            cand.append(float(tclk[kset + dd_[0]]))
        gg_ = np.where(gap_after[kset:ix[-1] + 1])[0]
        if len(gg_):
            cand.append(float(tclk[kset + gg_[0]]) + 0.2)
        t_dead = min(cand) if cand else np.inf
        t_end = min(t_next, t_dead, t_pend)
        if t_end <= t_set:
            continue
        f0 = gA.head(1)
        start = "recycle"
        if len(f0):
            r0 = f0.iloc[0]
            is_reg = ((r0.type == "ball_recovery" and not r0.recovery_failed) or
                      (r0.type == "interception" and r0.interception_outcome == "won") or
                      (r0.type == "duel" and r0.duel_type == "tackle" and r0.duel_outcome == "won"))
            if is_reg and (r0.location_x >= 60) and (t_set - r0.tt <= C.REGAIN_WINDOW):
                start = "regain"
        gE = gA[gA.tt <= t_end + C.EVENT_TOLERANCE]
        shots = gE[gE.type.eq("shot") & (gE.tt >= t_set)]
        sh_t = shots.tt.to_numpy()
        sh_xg = shots.xg.fillna(0.0).to_numpy()
        sg = sgn_of(A, p)
        base = {"match_id": mid, "season": season, "team_id": A, "opp_id": B, "period": p, "possession": int(n),
                "set_frame": int(fr[kset]), "t_set": t_set, "t_end": t_end, "dur": t_end - t_set,
                "play_pattern": pp, "start": start, "n_shots": len(sh_t), "xg": float(sh_xg.sum())}
        if light:
            ATT.append(base)
            for s_ in range(int(np.ceil(t_end - t_set))):
                k = frame_at(p, t_set + s_)
                if k >= 0:
                    STEPS.append({"match_id": mid, "season": season, "team_id": A, "opp_id": B, "possession": int(n),
                                  "period": p, "k": s_, "frame": int(fr[k]), "deep3_t": float(deep3_of[A][k])})
            continue
        # option events in this possession
        lb, lps = [], []
        for typ, store, completed in (("carry", lb, False), ("pass", lps, True)):
            q = gA[gA.type.eq(typ)]
            if completed:
                q = q[q.pass_outcome.eq("complete")]
            for r in q.itertuples():
                k0, k1 = frame_at(p, r.tt), frame_at(p, r.tt + r.dur)
                if k0 < 0 or k1 < 0:
                    continue
                Pk, b0, b1 = P[k0] * sg, bxy[k0] * sg, bxy[k1] * sg
                if np.isnan(b0).any() or np.isnan(b1).any():
                    continue
                ln = lines_at(Pk, goalside(Pk, b0, out_slots[B]))
                if ln is None:
                    continue
                if (half - b0[0] > ln[1]) and (half - b1[0] < ln[1]):   # starts in front of the midfield line, ends beyond
                    store.append(float(r.tt + r.dur))
        psA = gA[gA.type.eq("pass")]
        is_cross = psA.cross.to_numpy(bool)
        is_cut = psA.cut_back.to_numpy(bool)
        is_high = (psA.pass_height == "high").to_numpy()
        ps_t = psA.tt.astype(float).to_numpy()
        ld = ps_t[is_cross]
        lb, lps = np.array(lb), np.array(lps)
        t_pstart = float(g.tt.iloc[0])
        # what happens after the attack: next team on the ball, the opponent's next possession
        e_after = ev[(ev.period == p) & (ev.tt >= t_end - 0.05)]
        next_team = int(e_after.possession_team_id.iloc[0]) if len(e_after) else -1
        eb = e_after[e_after.possession_team_id == B]
        opp_xg20, loss_dist = 0.0, np.nan
        if len(eb):
            pe = eb[eb.possession == eb.possession.iloc[0]]
            sh = pe[pe.type.eq("shot") & (pe.team_id == B) & (pe.tt <= t_end + C.COUNTER_WINDOW)]
            opp_xg20 = float(sh.xg.fillna(0).sum())
            loc = pe[(pe.team_id == B) & pe.location_x.notna()]
            loss_dist = (120.0 - float(loc.location_x.iloc[0])) * Lp / 120.0 if len(loc) else np.nan
        ATT.append(base | {"t_pstart": t_pstart, "throw_in": pp == "from_throw_in",
                           "next_team": next_team, "opp_xg20": opp_xg20, "loss_dist": loss_dist})
        nst = int(np.ceil(t_end - t_set))
        step_t, step_kf = [], []
        for s_ in range(nst):
            t = t_set + s_
            k = frame_at(p, t)
            if k < 0:
                continue
            has_l, _, bm, mm_, ovl = state(A, k)
            mp = pidx[p]
            lo_, hi_ = np.searchsorted(tclk[mp], [max(t - C.WINDOW, t_pstart), t + 1e-6])
            wi = mp[lo_:hi_]
            by = bxy[wi[live[wi]], 1]
            yr = float(np.nanmax(by) - np.nanmin(by)) if len(by) and not np.isnan(by).all() else 0.0
            inw = lambda a: bool(((a >= t - C.WINDOW) & (a <= t)).any()) if len(a) else False
            w_ = (ps_t >= t - C.WINDOW) & (ps_t <= t)
            last = s_ == nst - 1
            o1 = lambda a: int(((a >= t) & ((a < t + 1) | last)).sum())
            STEPS.append({"match_id": mid, "season": season, "team_id": A, "opp_id": B, "possession": int(n),
                          "period": p, "t": t, "frame": int(fr[k]), "k": s_, "minute": float(t / 60 + 45 * (p - 1)),
                          "score_diff": score_diff(A, p, t), "start": start, "throw_in": pp == "from_throw_in",
                          "carry_into_block": inw(lb), "pass_into_block": inw(lps), "switch": yr >= C.SWITCH_M,
                          "cross": inw(ld), "cross_cutback": bool((w_ & is_cut).any()),
                          "cross_lofted": bool((w_ & is_cross & is_high).any()),
                          "extra_attacker": bool(ovl >= 1) if not np.isnan(ovl) else False, "overload": ovl,
                          "has_lines": has_l,
                          "ball_depth": float(half - bxy[k, 0] * sg), "ball_y": float(bxy[k, 1] * sg),
                          "deep3_t": float(deep3_of[A][k]),
                          "shots_1s": o1(sh_t), "xg_1s": float(sh_xg[(sh_t >= t) & ((sh_t < t + 1) | last)].sum())})
            step_t.append(t)
            step_kf.append(k)
            X, Mk = node_frame(A, p, k, 0)
            NT.append(X)
            MT.append(Mk)
        # decision moment t-5: the attack's first step at or after t-5, at the matching frame of its 5 s history
        step_t, step_kf = np.array(step_t), np.array(step_kf)
        for t in step_t:
            td = t - C.WINDOW
            j = int(np.searchsorted(step_t, td - 1e-4))
            back = int(np.clip(np.rint((step_t[j] - td) * 5), 0, 24))
            X, Mk = node_frame(A, p, int(step_kf[j]), back)
            ND.append(X)
            MD.append(Mk)
    inposs = {int(t): float(((poss == t) & live).sum() * 0.2) for t in (home, away)}
    out = {"mid": mid, "season": season, "attacks": pd.DataFrame(ATT), "steps": pd.DataFrame(STEPS),
           "inposs": inposs, "offsets": m["offsets"], "sync_dist": m["sync_dist"]}
    if not light:
        z = lambda L, s: np.stack(L) if L else np.zeros((0,) + s, np.float16 if len(s) == 2 else bool)
        out.update({"nodes_t": z(NT, (23, 4)), "mask_t": z(MT, (23,)), "nodes_dec": z(ND, (23, 4)), "mask_dec": z(MD, (23,))})
        if len(out["steps"]):
            Xt = out["nodes_t"].astype(np.float32)
            dfx = np.where(out["mask_t"][:, 12:22], Xt[:, 12:22, 0], np.nan)
            with np.errstate(all="ignore"):
                out["steps"]["deep3_nodes_t"] = np.nanmean(np.sort(dfx, axis=1)[:, :3], axis=1)
    return out
