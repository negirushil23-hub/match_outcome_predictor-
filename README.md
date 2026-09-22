# Premier League Match Outcome Predictor

Predicts whether a team will win, draw, or lose its next match using rolling form
stats and a RandomForest vs. XGBoost comparison.

## Project Structure

```
epl_predictor/
├── data/
│   ├── fetch_data.py        # pulls real EPL data from football-data.co.uk
│   ├── synthetic_data.py    # generates fake data for offline testing
│   └── raw_matches.csv      # (generated) raw match data
├── src/
│   ├── features.py          # rolling, leak-free feature engineering
│   ├── train_model.py       # trains + evaluates RandomForest & XGBoost
│   └── predict.py           # predicts a specific upcoming fixture
├── models/
│   └── best_model.joblib    # (generated) the better of the two models
├── outputs/
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_xgb.png
│   ├── feature_importance_xgb.png
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

## Setup

This was built and tested on macOS. A few things came up worth noting in case
you hit the same walls:

**1. `python`/`pip` not found at all.** macOS doesn't ship a usable `python`
command by default. Installed Python via Homebrew:

```bash
brew install python
```

**2. `python3 --version` showed 3.9.6 instead of the Homebrew install.** macOS
ships an old Python 3.9 with the Xcode Command Line Tools, and it was earlier
on PATH than the Homebrew one. Fixed by prepending Homebrew's Python to PATH
in `~/.zshrc`:

```bash
echo 'export PATH="/opt/homebrew/opt/python@3.14/libexec/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**3. `error: externally-managed-environment` on `pip install`.** Homebrew's
Python refuses global `pip install`s (PEP 668). The fix is a virtual
environment scoped to the project, not `--break-system-packages`:

```bash
python3 -m venv venv
source venv/bin/activate
```

Once the venv is active, `python` and `pip` (no `3`) both point at the venv's
interpreter, and installs are sandboxed to this project.

### Quick Start

```bash
cd epl_predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 1: get data. Either:
python data/fetch_data.py --seasons 10        # real data, needs internet
# or, to test the pipeline without internet:
python data/synthetic_data.py                 # fake data, dry-run only

# Step 2: train
python src/train_model.py

# Step 3: predict a fixture
python src/predict.py --team "Arsenal" --opponent "Chelsea" --venue home
```

Remember to run `source venv/bin/activate` again each time you open a new
terminal for this project. `deactivate` exits it.

## How the Data Is Shaped

Each match produces **two rows**, one from each team's perspective
(`is_home` = 1 or 0), instead of one row per fixture. That's what lets the
model answer "will *this team* win its *next* match" rather than only "who
wins *this specific fixture*" from a fixed home/away angle.

For every row, each rolling stat (goals scored/conceded, shots, shots on
target, possession, recent-form points) is computed **only from that team's
matches strictly before the one being predicted** — `features.py` shifts by
one match before applying the rolling window. This is the detail that
matters most in the whole pipeline: skip it and the model trains on stats
that already contain the outcome it's supposed to predict, and every
accuracy number that follows is meaningless.

| Feature | Window | Notes |
|---|---|---|
| `goals_scored_per90` | last 10 matches | mean, pre-match |
| `goals_conceded_per90` | last 10 matches | mean, pre-match |
| `shots_avg` | last 10 matches | mean, pre-match |
| `shots_on_target_avg` | last 10 matches | mean, pre-match |
| `possession_avg` | last 10 matches | see note below |
| `is_home` | — | 1 / 0, captures home advantage |
| `form_pts_last5` | last 5 matches | 3/1/0 points per W/D/L, summed |

**Possession note:** football-data.co.uk, the free source `fetch_data.py`
pulls from, doesn't track possession — that's an Opta/FBref-level stat. The
pipeline imputes 50% when it's missing and adds a `possession_missing` flag
so the model can learn to discount it rather than being fed a silently wrong
number. For real possession data, the cleanest free route is scraping FBref
with the `soccerdata` package and joining it onto `raw_matches.csv` by date
and team name before running `features.py`.

## Why a Chronological Split, Not Random

`train_model.py` sorts by date and holds out the most recent 20% as the test
set, instead of a random `train_test_split`. Match data is a time series —
a random shuffle would let the model train on a team's April form while
being tested on its March match, which inflates accuracy in a way that
wouldn't hold up on genuinely future fixtures.

## Model Notes

- **RandomForestClassifier**: `class_weight="balanced"`, since draws are a
  minority class (roughly a quarter of matches) and get ignored otherwise.
- **XGBClassifier**: `multi:softprob`, shallow trees (`max_depth=4`) and a
  low learning rate to resist overfitting on a single-league dataset.
- Expect roughly 50–55% accuracy against a ~38–45% majority-class baseline.
  Draws will be the hardest class to call — bookmakers and analysts have the
  same problem, so weak precision/recall on `D` is the expected shape of
  this problem, not a bug in the code.

## Extending This

- **More seasons** stabilizes the rolling stats earlier in each season —
  `fetch_data.py --seasons 15` pulls further back.
- **Head-to-head features**: a rolling "form vs. this specific opponent"
  feature would slot into `features.py`.
- **Odds as a feature**: football-data.co.uk also has bookmaker odds columns
  (`B365H/D/A` etc.) — strong predictors on their own and a good benchmark
  to try to beat.
- **Hyperparameter tuning**: wrap `train_random_forest` / `train_xgboost` in
  `GridSearchCV` or `optuna` once running on real data.
