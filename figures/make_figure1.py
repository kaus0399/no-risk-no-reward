"""Draw Figure 1 from results/abstract_results.json.

(a) The options at a wide decision against a set low block (schematic; players are illustrative).
(b) Risk and reward of each option against not taking it, pooled over 2024/25 and 2025/26.

Usage: python figures/make_figure1.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402
from mplsoccer import VerticalPitch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "abstract_results.json").read_text())["figure_1b_each_option_vs_not_taking_it"]

PITCH, PITCH_ALT = "#4b6655", "#47614f"
ATT, DEF, BALL = "white", "#2b2b40", "#f2c200"
SWITCH, RECYCLE = "#a8d8f0", "#d9d9d9"
GREEN, ORANGE = "#2c6e49", "#d9822b"
STROKE = [pe.withStroke(linewidth=2.8, foreground="#1d2b22")]

# Schematic coordinates: metres across from the centre line (+ = ball side), metres from the goal line.
BALL_AT = (26, 24.5)
BACK_FOUR = [(-9, 11), (-1.5, 10.5), (6, 10.5), (13.5, 12)]
MIDFIELD_FOUR = [(-8.5, 18.5), (0.5, 17.5), (10.5, 17.2), (25.5, 20.3)]
FORWARDS = [(-3, 32.5), (8, 31.5)]
KEEPER = (1, 2.5)
TEAM = {"extra attacker": (27.5, 31), "striker": (-3, 12.8), "far post": (-11.5, 14), "pocket": (-2, 15.5),
        "far winger": (-27, 26), "back": (16, 37.5), "mid": (-7, 38.5)}
# The drawing obeys the set-low-block rule: deepest three average within 25 m, 8+ outfield players goal-side.
assert sum(sorted(y for _, y in BACK_FOUR + MIDFIELD_FOUR + FORWARDS)[:3]) / 3 <= 25
assert sum(y < BALL_AT[1] for _, y in BACK_FOUR + MIDFIELD_FOUR + FORWARDS) >= 8


def T(x_across, y_goal):
    """Schematic coordinates -> axes coordinates of a vertical half pitch with the goal at the top."""
    return 34 - x_across, 105 - y_goal


def arrow(ax, a, b, color, ls="-", rad=0.0, lw=1.8, shrinkB=4):
    ax.add_patch(FancyArrowPatch(T(*a), T(*b), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, ls=ls, shrinkA=3, shrinkB=shrinkB, zorder=5))


def panel_a(ax, ms=7, fs=10.5):
    VerticalPitch(pitch_type="custom", pitch_length=105, pitch_width=68, half=True, pitch_color=PITCH, stripe=True,
                  stripe_color=PITCH_ALT, line_color="white", linewidth=1.4, pad_bottom=1).draw(ax=ax)
    for x0 in (20.16, -34):  # wide zones: outside the penalty-area width, 12-35 m from goal
        X, Y = T(x0 + 13.84, 35)
        ax.add_patch(Rectangle((X, Y), 13.84, 23, color="#cfe8ff", alpha=0.28, lw=0, zorder=1))
    (xa, ya), (xb, yb) = T(-34, 25), T(34, 25)
    ax.plot([xa, xb], [ya, yb], color="white", lw=1.0, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.text(*T(-20.5, 24.4), "25 m", fontsize=10, color="white", ha="left", va="bottom", path_effects=STROKE)
    for x, y in BACK_FOUR + MIDFIELD_FOUR + FORWARDS + [KEEPER]:
        ax.plot(*T(x, y), "o", color=DEF, mec="white", mew=1.0, ms=ms, zorder=3)
    for x, y in list(TEAM.values()) + [BALL_AT]:
        ax.plot(*T(x, y), "o", color=ATT, mec="black", mew=1.0, ms=ms, zorder=4)
    circ = Circle(T(*BALL_AT), 10, fill=False, ec="white", lw=1.3, ls=(0, (1.5, 2)), zorder=2)
    ax.add_patch(circ)
    circ.set_clip_path(Rectangle((0, 52.5), 68, 52.5, transform=ax.transData))
    arrow(ax, BALL_AT, (-1, 6.5), BALL, rad=0.33, lw=2.0, shrinkB=1)            # cross
    arrow(ax, TEAM["striker"], (-2, 7.5), ATT, ls=(0, (3, 2)), lw=1.3)         # striker's run
    arrow(ax, BALL_AT, TEAM["pocket"], ATT, lw=1.8)                             # pass into the block
    arrow(ax, BALL_AT, (18.5, 16.2), ATT, ls=(0, (4, 2.5)), lw=2.0, shrinkB=1)  # carry into the block
    arrow(ax, BALL_AT, TEAM["far winger"], SWITCH, rad=-0.14, lw=1.8)           # switch
    arrow(ax, BALL_AT, TEAM["back"], RECYCLE, lw=1.8)                           # recycle
    ax.plot(*T(BALL_AT[0] + 1.0, BALL_AT[1] - 1.0), "o", color=BALL, mec="black", mew=0.8, ms=5.5, zorder=6)
    lab = dict(fontsize=fs, fontweight="bold", zorder=7, path_effects=STROKE)
    ax.text(*T(19, 3.0), "cross (lofted or low)", color=BALL, ha="center", va="center", **lab)
    ax.text(*T(3.0, 19.6), "pass", color="white", ha="center", va="center", **lab)
    ax.text(*T(15.2, 15.6), "carry", color="white", ha="center", va="center", **lab)
    ax.text(*T(-27, 28.0), "switch", color=SWITCH, ha="center", va="top", **lab)
    ax.text(*T(16, 40.0), "recycle", color=RECYCLE, ha="center", va="top", **lab)
    ax.text(*T(29.5, 33.0), "extra\nattacker", color="white", ha="center", va="top", linespacing=1.0, **lab)
    ax.text(*T(33.3, 12.6), "10 m", color="white", ha="right", va="top", fontsize=10, zorder=7, path_effects=STROKE)
    ax.text(0.0, 1.01, "(a)", transform=ax.transAxes, fontsize=12, va="bottom")


def panel_b(ax):
    labels = {"pass into the block": ("pass", (9, 0), "left", GREEN),
              "carry into the block": ("carry", (-9, 0), "right", GREEN),
              "switch": ("switch", (9, -2), "left", GREEN),
              "extra attacker near the ball": ("extra\nattacker", (9, 0), "left", ORANGE),
              "lofted cross": ("lofted cross", (-9, 0), "right", GREEN),
              "low cross": ("low cross", (-9, 0), "right", GREEN)}
    ax.set_xscale("log", base=2)
    ax.axhline(1, color="0.6", lw=0.9)
    ax.axvline(1, color="0.6", lw=0.9)
    ax.add_patch(Rectangle((0.25, 1.0), 0.75, 0.8, color="0.93", lw=0, zorder=0))
    ax.text(0.5, 1.68, "safer and more\nvaluable: none", ha="center", va="top", fontsize=11.5, color="0.4")
    ax.text(1.04, 0.985, "not taking\nthe option", ha="left", va="top", fontsize=10.5, color="0.4", linespacing=1.0)
    for key, (lab, off, ha, col) in labels.items():
        x, y = RES[key]["risk_turnover_odds_ratio"], RES[key]["reward_net_xg_ratio"]
        assert (x[1] > 1 or x[2] < 1) and (y[1] > 1 or y[2] < 1), key  # every 95% interval excludes 1
        ax.plot(x[0], y[0], "o", color=col, ms=9, mec="white", mew=1.2, zorder=3)
        ax.annotate(lab, (x[0], y[0]), xytext=off, textcoords="offset points", fontsize=11.5, ha=ha, va="center",
                    linespacing=1.05)
    ax.set_xlim(0.25, 4.6)
    ax.set_ylim(0.8, 1.8)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×"])
    ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_yticklabels(["0.8×", "1×", "1.2×", "1.4×", "1.6×", "1.8×"])
    ax.minorticks_off()
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel("Risk: losing the ball  →", fontsize=12.5)
    ax.set_ylabel("Reward: net xG  →", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.16, 1.03, "(b)", transform=ax.transAxes, fontsize=13, va="bottom")


def main():
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(10.5, 4.6))
    panel_a(fig.add_axes([0.0, 0.02, 0.50, 0.92]))
    panel_b(fig.add_axes([0.6, 0.15, 0.385, 0.76]))
    out = ROOT / "figures" / "figure1.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
