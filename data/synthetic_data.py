"""
synthetic_data.py
------------------
Generates a plausible fake Premier League match dataset so the rest of
the pipeline (feature engineering, training, evaluation) can be built
and tested without live internet access. NOT for real predictions.
"""

import numpy as np
import pandas as pd

TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
    "Man City", "Man United", "Newcastle", "Nottingham Forest",
    "Tottenham", "West Ham", "Wolves", "Leicester", "Southampton", "Ipswich",
]

# Rough relative strength so results aren't pure noise.
rng = np.random.default_rng(42)
STRENGTH = {t: rng.normal(0, 1) for t in TEAMS}
STRENGTH.update({"Man City": 2.2, "Arsenal": 1.8, "Liverpool": 1.9,
                  "Chelsea": 1.2, "Tottenham": 1.1, "Man United": 1.0,
                  "Newcastle": 1.0})


def generate(n_seasons=6, start_year=2019, matches_per_season_per_team=38, out_path=None):
    rows = []
    for s in range(n_seasons):
        season_start = start_year + s
        season = f"{season_start}-{season_start+1}"
        # simple round-robin-ish schedule (not exact fixture list, fine for testing)
        fixtures = []
        for home in TEAMS:
            for away in TEAMS:
                if home != away:
                    fixtures.append((home, away))
        rng.shuffle(fixtures)
        fixtures = fixtures[: matches_per_season_per_team * len(TEAMS) // 2]

        base_date = pd.Timestamp(f"{season_start}-08-10")
        for i, (home, away) in enumerate(fixtures):
            date = base_date + pd.Timedelta(days=int(i * 2.3))
            home_strength = STRENGTH[home] + 0.25  # home advantage bump
            away_strength = STRENGTH[away]
            diff = home_strength - away_strength

            lam_home = max(0.3, 1.35 + diff * 0.35)
            lam_away = max(0.3, 1.15 - diff * 0.30)
            fthg = rng.poisson(lam_home)
            ftag = rng.poisson(lam_away)

            if fthg > ftag:
                ftr = "H"
            elif fthg < ftag:
                ftr = "A"
            else:
                ftr = "D"

            hs = max(1, int(rng.normal(13 + diff * 2, 4)))
            as_ = max(1, int(rng.normal(11 - diff * 2, 4)))
            hst = max(0, int(hs * rng.uniform(0.3, 0.5)))
            ast = max(0, int(as_ * rng.uniform(0.3, 0.5)))
            hc = max(0, int(rng.normal(5, 2)))
            ac = max(0, int(rng.normal(4, 2)))
            poss_home = int(np.clip(50 + diff * 6 + rng.normal(0, 5), 25, 75))
            poss_away = 100 - poss_home

            rows.append({
                "Date": date, "HomeTeam": home, "AwayTeam": away,
                "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
                "HS": hs, "AS": as_, "HST": hst, "AST": ast,
                "HC": hc, "AC": ac,
                "HomePossession": poss_home, "AwayPossession": poss_away,
                "Season": season,
            })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    if out_path:
        df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = generate(out_path="data/raw_matches.csv")
    print(f"Generated {len(df)} synthetic matches, saved to data/raw_matches.csv")
    print(df.head())
