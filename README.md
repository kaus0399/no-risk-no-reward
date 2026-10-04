# No Risk, No Reward

**Pricing Attacking Options Against the Low Block in Soccer with Graph Attention Networks**

Code and results for our MIT Sloan Sports Analytics Conference 2027 research paper submission.

![Figure 1](figures/figure1.png)

*Figure 1. (a) The options at a wide decision against a set low block (illustrative players). (b) Each option against not taking it, at all moments against a set low block, as ratios; every 95% interval excludes 1.*

## Overview

When a team attacks a set low block, each choice is a trade. A cross or a pass into the block can open a chance and can also lose the ball. We put a price on each option in expected goals (xG):

- **Reward:** xG created over the rest of the attack.
- **Risk:** losing the ball within 5 s, and xG conceded on the counter within 20 s.

Options are not taken at random: teams cross when crossing is on. A graph attention network (GAT) reads the 22 players and the ball at each decision and estimates how likely every option was. Doubly robust estimation then compares decisions where an option was equally likely given everything the model sees, which removes the advantage an option gets from being chosen in good moments.

## Getting started

With Python 3.12:

```
pip install -r requirements.txt
```

The pipeline runs in seven steps. `DATA` is the folder with your match data (see [Data](#data)); it can also be set with the `LOWBLOCK_DATA_ROOT` environment variable. Intermediate tables go to the folder given by `--out`, and the reported numbers to `results/`.

| Step | Command | Output |
|---|---|---|
| 1. Decision steps against a set low block | `python scripts/01_build_steps.py --data-root DATA --out derived` | one row per second of each attack |
| 2. Team strength | `python scripts/02_team_strength.py --data-root DATA --out derived` | strength per team and match |
| 3. Graph inputs for the choice model | `python scripts/03_choice_inputs.py --out derived` | positions, velocities and context per decision |
| 4. GAT choice model | `python scripts/04_train_choice_model.py --out derived` | out-of-season choice probabilities, `results/choice_model.json` |
| 5. Prices of the wide decision | `python scripts/05_price_options.py --out derived` | `results/table1.json` (Table 1) |
| 6. Each option against not taking it | `python scripts/06_option_ratios.py --out derived` | `results/figure1b.json` (Figure 1b) |
| 7. Exposure to set low blocks | `python scripts/07_low_block_share.py --out derived` | `results/low_block_share.json` |

If your files use different column names, add `--schema FILE` to steps 1 and 2. For the trend across seasons in step 7, first run step 1 with `--light --seasons` and the seasons to include; on our tracking data from 2021/22 to 2025/26 the share is steady. On a laptop, step 1 takes about 15 to 30 minutes for 760 matches with four workers, step 4 about 25 minutes, and step 5 about 25 minutes.

Figure 1 is drawn from the files in `results/` and needs no match data:

```
python figures/make_figure1.py
```

## Data

The code expects tracking data with the fields of [kloppy](https://kloppy.pysport.org)'s tracking model and event data with the core fields of [SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html) (Decroos et al., 2019), plus possession number, pass height and shot xG.

The 760 Premier League matches (2024/25 and 2025/26) behind our results are proprietary and cannot be released. The pipeline can be applied to any other dataset prepared in the same format, with each match in its own folder:

```
DATA/<season>/<match>/
    match_metadata.json
    tracking/tracking.parquet
    events/events.csv
```

Season folders are named like `20242025`, and the seasons used are set in `lowblock/config.py`. Team and player ids must match across tracking and events. Fields, units and accepted values are listed in `lowblock/schema.py`, and a mapping file passed with `--schema` adapts other column names. Event times are aligned with the tracking clock automatically. The aggregate results we report are in `results/`.

## Pipeline in detail

**Set low block (step 1).** We build on [SkillCorner's phase definition](https://skillcorner.com/us/articles/game-intelligence-out-of-possession), in which the average position of the defending team's deepest three players is within its defensive third. Our stricter version requires the three deepest outfield defenders to average within 25 m of their goal, with at least eight outfield defenders goal-side of the ball, for 3 s of open play at 11 v 11. A decision counts only while the block is still within 25 m.

**Options (step 1).** For each second of an attack against a set low block, the option is what the team does in the next 5 s:

| Option | Definition |
|---|---|
| Cross | an event marked as a cross; lofted if its pass height is high, otherwise low (cut-backs are included in the cross share of decisions but are too few to price on their own) |
| Pass or carry into the block | the ball goes beyond the block's midfield line |
| Switch | the ball moves at least 20 m across the pitch |
| Extra attacker | more attackers than defenders within 10 m of the ball |
| Recycle | no cross and no pass or carry into the block |

A **wide decision** is one where the ball is outside the width of the penalty area, 12 to 35 m from goal.

**Outcomes (step 1).** xG over the rest of the attack; whether the ball is lost within 5 s; opponent xG within 20 s after the attack ends. Net xG is xG created minus counter xG conceded.

**Team strength (step 2).** Each team's season-average non-penalty xG difference per match, leaving out the match itself. It is used as a control for both teams.

**Choice model (steps 3 and 4).** `lowblock/choice_model.py` is a dense GATv2 network over the 23 nodes (players and ball), with edge features from pairwise distances, followed by a small head that also takes seven context values (minute, score, both teams' strength, regain, throw-in, time since the block set). It is trained on one season and applied to the other, so every probability is out of season. On the held-out season it predicts crosses with an AUC of 0.79, against 0.68 for a model that uses ball position alone.

**Prices at equal likelihood (step 5).** `lowblock/aipw.py` combines the choice probabilities with outcome models fitted by gradient boosting (five folds, split by match). Decisions where either option had a probability below 0.02 are left out. Prices are differences against recycling per 100 five-second decisions, with 95% intervals from a bootstrap that resamples whole matches.

**Each option against not taking it (step 6).** Regressions of losing the ball (logistic) and of net xG on each option, adjusted for ball position, score, minute, both teams' strength, time since the block set and how the possession started, with match-clustered standard errors.

## Examples

`examples/selection_demo.py` shows, without any match data, why options have to be compared at equal likelihood. It simulates decisions in which teams cross or play into the block more often when a chance is already likely, so the true value of each option is known. Averaged over five simulations:

| Option | True value | Raw comparison | Doubly robust | Intervals containing the true value |
|---|---|---|---|---|
| Pass or carry into the block | 0.30 | 0.84 | 0.32 | 5 of 5 |
| Cross | 0.85 | 1.80 | 0.87 | 4 of 5 |

Values are xG per 100 decisions against recycling. The raw comparison overstates both options; the doubly robust estimator in `lowblock/aipw.py` recovers them.

```
python examples/selection_demo.py
```

## Repository layout

```
lowblock/        package: data loading, low-block detection, steps, models, estimators
scripts/         the seven pipeline steps
examples/        a simulation that runs without match data
figures/         Figure 1 and the script that draws it
results/         aggregate results reported in the paper
```

## License

MIT
