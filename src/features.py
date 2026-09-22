"""
features.py
------------
Turns raw match-level rows (one row per match, home & away columns
side by side) into a TEAM-CENTRIC table: one row per team per match,
with rolling-window features computed ONLY from that team's past
matches (so there is no leakage from the match being predicted).

Each match produces two rows: one from the home team's perspective,
one from the away team's. This is what lets the final model answer
"will TEAM X win/draw/lose its NEXT match", which is how the brief
frames the prediction target, rather than only "who wins this
fixture" from a fixed home/away angle.

Features per team, computed as of just BEFORE the match:
    goals_scored_per90       - rolling mean goals scored (last N matches)
    goals_conceded_per90     - rolling mean goals conceded (last N matches)
    shots_avg                - rolling mean shots taken (last N matches)
    shots_on_target_avg      - rolling mean shots on target (last N matches)
    possession_avg           - rolling mean possession % (last N matches)
    is_home                  - 1 if this row is the team's home fixture
    form_pts_last5           - points won in the team's last 5 matches

Target:
    result  ->  'W' / 'D' / 'L'  for that team in that match
"""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 10       # matches used for the per90-style rolling averages
FORM_WINDOW = 5           # matches used for the recent form points feature


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-match into two-rows-per-match (team-centric)."""
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"])

    has_poss = "HomePossession" in matches.columns and "AwayPossession" in matches.columns

    home_rows = pd.DataFrame({
        "Date": matches["Date"],
        "Season": matches["Season"],
        "team": matches["HomeTeam"],
        "opponent": matches["AwayTeam"],
        "is_home": 1,
        "goals_scored": matches["FTHG"],
        "goals_conceded": matches["FTAG"],
        "shots": matches["HS"],
        "shots_on_target": matches["HST"],
        "possession": matches["HomePossession"] if has_poss else np.nan,
        "result": np.where(matches["FTR"] == "H", "W",
                    np.where(matches["FTR"] == "D", "D", "L")),
    })
    away_rows = pd.DataFrame({
        "Date": matches["Date"],
        "Season": matches["Season"],
        "team": matches["AwayTeam"],
        "opponent": matches["HomeTeam"],
        "is_home": 0,
        "goals_scored": matches["FTAG"],
        "goals_conceded": matches["FTHG"],
        "shots": matches["AS"],
        "shots_on_target": matches["AST"],
        "possession": matches["AwayPossession"] if has_poss else np.nan,
        "result": np.where(matches["FTR"] == "A", "W",
                    np.where(matches["FTR"] == "D", "D", "L")),
    })

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date"]).reset_index(drop=True)
    return long_df


def _points(result: str) -> int:
    return {"W": 3, "D": 1, "L": 0}[result]


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Takes the raw match dataframe (one row per match,
    schema matching fetch_data.py/synthetic_data.py output) and
    returns a team-centric feature table ready for model training.
    """
    long_df = _long_format(matches)
    long_df["points"] = long_df["result"].map(_points)

    feature_frames = []
    for team, grp in long_df.groupby("team", sort=False):
        grp = grp.sort_values("Date").copy()

        # shift(1) is the leakage guard: every rolling stat is computed
        # using only matches strictly BEFORE the current one.
        prior_goals_scored = grp["goals_scored"].shift(1)
        prior_goals_conceded = grp["goals_conceded"].shift(1)
        prior_shots = grp["shots"].shift(1)
        prior_shots_on_target = grp["shots_on_target"].shift(1)
        prior_possession = grp["possession"].shift(1)
        prior_points = grp["points"].shift(1)

        grp["goals_scored_per90"] = prior_goals_scored.rolling(ROLLING_WINDOW, min_periods=3).mean()
        grp["goals_conceded_per90"] = prior_goals_conceded.rolling(ROLLING_WINDOW, min_periods=3).mean()
        grp["shots_avg"] = prior_shots.rolling(ROLLING_WINDOW, min_periods=3).mean()
        grp["shots_on_target_avg"] = prior_shots_on_target.rolling(ROLLING_WINDOW, min_periods=3).mean()
        grp["possession_avg"] = prior_possession.rolling(ROLLING_WINDOW, min_periods=3).mean()
        grp["form_pts_last5"] = prior_points.rolling(FORM_WINDOW, min_periods=1).sum()

        feature_frames.append(grp)

    feat_df = pd.concat(feature_frames, ignore_index=True)

    # Drop early season rows where rolling stats aren't reliable yet
    feat_df = feat_df.dropna(subset=[
        "goals_scored_per90", "goals_conceded_per90",
        "shots_avg", "shots_on_target_avg", "form_pts_last5",
    ])

    # Possession may be entirely missing (real football-data.co.uk has no
    # possession column)
    feat_df["possession_missing"] = feat_df["possession_avg"].isna().astype(int)
    feat_df["possession_avg"] = feat_df["possession_avg"].fillna(50.0)

    feat_df = feat_df.sort_values("Date").reset_index(drop=True)
    return feat_df


FEATURE_COLS = [
    "goals_scored_per90", "goals_conceded_per90",
    "shots_avg", "shots_on_target_avg",
    "possession_avg", "possession_missing",
    "is_home", "form_pts_last5",
]
TARGET_COL = "result"


if __name__ == "__main__":
    raw = pd.read_csv("data/raw_matches.csv")
    feats = build_features(raw)
    feats.to_csv("data/team_match_features.csv", index=False)
    print(f"Built {len(feats)} team-match feature rows")
    print(feats[FEATURE_COLS + [TARGET_COL]].head())
