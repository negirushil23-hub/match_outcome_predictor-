"""
predict.py
----------
Loads the trained model and predicts the outcome (W/D/L) for a team's
next match, given the opponent and venue. Uses each team's most recent
rolling stats (from the historical data) as the feature snapshot.

Usage:
    python src/predict.py --team "Arsenal" --opponent "Chelsea" --venue home
"""

import argparse

import joblib
import pandas as pd

from features import FEATURE_COLS, build_features


def latest_team_snapshot(feat_df: pd.DataFrame, team: str, is_home: int) -> pd.DataFrame:
    """
    Grab the most recent rolling-feature row we have for this team, and
    overwrite is_home with the venue of the match being predicted (the
    other rolling stats -- form, goals per90, etc. -- carry over as the
    team's current level of form regardless of where the next game is).
    """
    team_rows = feat_df[feat_df["team"] == team].sort_values("Date")
    if team_rows.empty:
        raise ValueError(f"No historical data found for team '{team}'. "
                          f"Check the spelling matches the dataset exactly.")
    snapshot = team_rows.iloc[[-1]][FEATURE_COLS].copy()
    snapshot["is_home"] = is_home
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True, help="Team to predict for")
    parser.add_argument("--opponent", required=True, help="Opponent (used for context in output only)")
    parser.add_argument("--venue", choices=["home", "away"], required=True)
    parser.add_argument("--model_path", default="models/best_model.joblib")
    parser.add_argument("--data_path", default="data/raw_matches.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    model, encoder, model_type = bundle["model"], bundle["encoder"], bundle["model_type"]

    raw = pd.read_csv(args.data_path)
    feat_df = build_features(raw)

    is_home = 1 if args.venue == "home" else 0
    snapshot = latest_team_snapshot(feat_df, args.team, is_home)

    if model_type == "xgboost":
        probs = model.predict_proba(snapshot)[0]
    else:
        probs = model.predict_proba(snapshot)[0]

    label_order = list(encoder.classes_)
    result = dict(zip(label_order, probs))

    print(f"\nPrediction: {args.team} ({args.venue}) vs {args.opponent}")
    print("-" * 45)
    for label, name in zip(["W", "D", "L"], ["Win", "Draw", "Loss"]):
        print(f"  {name:5s}: {result.get(label, 0):.1%}")

    predicted = max(result, key=result.get)
    label_names = {"W": "WIN", "D": "DRAW", "L": "LOSS"}
    print(f"\n  => Most likely outcome for {args.team}: {label_names[predicted]}")


if __name__ == "__main__":
    main()
