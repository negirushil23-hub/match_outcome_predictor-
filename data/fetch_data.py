"""
fetch_data.py
--------------
Pulls historical Premier League match data from football-data.co.uk.

Run this on a machine with normal internet access (it is NOT run
inside the sandbox that built this project, since that environment's
network is locked down to package registries only).

Usage:
    python fetch_data.py                # pulls last 10 seasons (default)
    python fetch_data.py --seasons 15    # pulls last 15 seasons
    python fetch_data.py --out data/raw_matches.csv
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

# Columns we actually need. football-data.co.uk has changed its column
# set slightly over the years, so we select defensively and fill
# missing optional columns with NaN.
KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # full-time home/away goals, result
    "HS", "AS",                     # shots
    "HST", "AST",                   # shots on target
    "HC", "AC",                     # corners (not used yet, kept for future features)
]


def season_code(start_year: int) -> str:
    """2023 -> '2324' (football-data.co.uk's season naming convention)."""
    yy1 = str(start_year)[-2:]
    yy2 = str(start_year + 1)[-2:]
    return f"{yy1}{yy2}"


def fetch_one_season(start_year: int) -> pd.DataFrame:
    code = season_code(start_year)
    url = BASE_URL.format(season_code=code)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df["Season"] = f"{start_year}-{start_year + 1}"
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=10,
                         help="How many most-recent completed seasons to pull")
    parser.add_argument("--end_year", type=int, default=2024,
                         help="Start-year of the most recent season to include "
                              "(2024 = the 2024-25 season)")
    parser.add_argument("--out", type=str, default="data/raw_matches.csv")
    args = parser.parse_args()

    frames = []
    for i in range(args.seasons):
        start_year = args.end_year - i
        try:
            print(f"Fetching {start_year}-{start_year+1} season...")
            frames.append(fetch_one_season(start_year))
            time.sleep(0.5)  # be polite to the server
        except Exception as e:
            print(f"  skipped {start_year}: {e}", file=sys.stderr)

    if not frames:
        print("No data fetched. Check your internet connection / URL format.")
        sys.exit(1)

    all_matches = pd.concat(frames, ignore_index=True)
    all_matches["Date"] = pd.to_datetime(all_matches["Date"], dayfirst=True, errors="coerce")
    all_matches = all_matches.dropna(subset=["Date", "FTR"]).sort_values("Date")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_matches.to_csv(out_path, index=False)
    print(f"Saved {len(all_matches)} matches across {all_matches['Season'].nunique()} seasons to {out_path}")


if __name__ == "__main__":
    main()
