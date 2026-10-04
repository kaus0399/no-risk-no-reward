"""Thresholds, windows and seeds used throughout the pipeline."""
import os
from pathlib import Path

# Set low block
DEEP3_MAX = 25.0            # mean depth of the three deepest outfield defenders, metres from their goal line
N_GOALSIDE = 8              # outfield defenders closer to the goal centre than the ball
SET_FRAMES = 15             # condition held for 3 s (5 Hz frames) before the block counts as set
FLICKER_FRAMES = 5          # breaks shorter than 1 s are ignored while the block sets
DEAD_FRAMES = 5             # ball dead for 1 s or more ends the attack
SET_PIECE_WINDOW = 10.0     # open play only: not within 10 s after a corner or a free kick
REGAIN_WINDOW = 10.0        # a possession that starts with a regain within 10 s of the block setting
EVENT_TOLERANCE = 1.0       # events up to 1 s after the end of an attack still count for it

# Options and outcomes
WINDOW = 5.0                # option window [t-5, t] and the 5 s turnover window, seconds
COUNTER_WINDOW = 20.0       # opponent xG within 20 s after the attack ends
SWITCH_M = 20.0             # ball moves at least 20 m across within the window
OVERLOAD_R = 10.0           # extra attacker: more attackers than defenders within 10 m of the ball
MIN_LINE = 3                # each defensive line needs at least three players

# Wide decision: ball outside the width of the penalty area, 12-35 m from goal (at the decision moment)
BOX_HALF_WIDTH = 20.16
WIDE_DEPTH = (12.0, 35.0)

# Tracking
FPS = 25
STEP = 5                    # use every 5th frame (5 Hz)
VMAX_PLAYER, VMAX_BALL = 12.0, 40.0

# Choice and outcome models
TRIM = 0.02                 # leave out decisions where either option has probability below 0.02
N_FOLDS = 5
N_BOOT = 1000               # bootstrap resamples (pooled seasons)
N_BOOT_SEASON = 500         # bootstrap resamples per season
SEED = 7070                 # folds and gradient-boosting models
SEED_GAT = 7071             # graph model training and its validation matches
SEED_AUC = 7072             # bootstrap of the held-out AUC
SEED_LOSS = 6609            # bootstrap of where the ball is lost
SEED_SHARE = 6607           # team-season bootstrap of the pooled low-block share
SEED_SHARE_SEASON = 6610    # team-season bootstrap per season (trend)

SEASONS = ["20242025", "20252026"]
TREND_SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]


def data_root(arg=None):
    """Data root from the command line, else the LOWBLOCK_DATA_ROOT environment variable."""
    root = arg or os.environ.get("LOWBLOCK_DATA_ROOT")
    if not root:
        raise SystemExit("set --data-root or LOWBLOCK_DATA_ROOT")
    return Path(root)
