"""
src/predictor.py
Fetches today's MLB schedule, lineups, and probable starters via the MLB
Stats API (statsapi package), then runs the trained Random Forest to rank
hitters by their home-run probability.
"""

import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import statsapi

from src.ballparks import get_park, is_dome
from src.weather import get_weather
from src.features import FeatureBuilder, _league_avg_pitcher
from src.model import HRModel
from src.data_fetcher import DataFetcher
from config import CURRENT_YEAR, MIN_PA_CURRENT_SEASON, TOP_N


class Predictor:
    def __init__(self, fetcher: DataFetcher):
        self.fetcher  = fetcher
        self.builder  = FeatureBuilder()
        self.today    = date.today().isoformat()

    # ════════════════════════════════════════════════════════════════════
    # PUBLIC
    # ════════════════════════════════════════════════════════════════════

    def predict_today(self, model: HRModel) -> list[dict]:
        """
        Main entry point.  Returns a list of prediction dicts sorted by
        hr_probability descending.
        """
        print(f"\nFetching today's schedule ({self.today})…")
        games = self._get_todays_games()
        if not games:
            print("No MLB games scheduled today.")
            return []

        print(f"Found {len(games)} games.")

        # Load current-season stats once
        bat_df  = self.fetcher.get_batting_stats(CURRENT_YEAR)
        pit_df  = self.fetcher.get_pitching_stats(CURRENT_YEAR)
        tpit_df = self.fetcher.get_team_pitching(CURRENT_YEAR)

        predictions = []
        for game in games:
            try:
                rows = self._predict_game(game, model, bat_df, pit_df, tpit_df)
                predictions.extend(rows)
            except Exception as exc:
                print(f"  [error] game {game.get('game_id')}: {exc}")

        if not predictions:
            return []

        predictions.sort(key=lambda r: r["hr_probability"], reverse=True)
        return predictions[:TOP_N]

    # ════════════════════════════════════════════════════════════════════
    # GAME LOOP
    # ════════════════════════════════════════════════════════════════════

    def _predict_game(
        self,
        game: dict,
        model: HRModel,
        bat_df: pd.DataFrame,
        pit_df: pd.DataFrame,
        tpit_df: pd.DataFrame,
    ) -> list[dict]:

        game_id    = game["game_id"]
        home_team  = game["home_name"]
        away_team  = game["away_name"]
        home_abbr  = game.get("home_abbr", "")
        away_abbr  = game.get("away_abbr", "")
        game_time  = game.get("game_datetime", "")
        venue      = game.get("venue_name", home_team)

        print(f"\n  {away_team} @ {home_team}  [{game_time[:10] if game_time else ''}]")

        # Park & weather
        park      = get_park(home_abbr)
        dome      = is_dome(home_abbr)
        weather   = get_weather(park["lat"], park["lon"]) if not dome else None

        if weather:
            print(f"    Weather: {weather['temp_f']}°F  "
                  f"Wind {weather['wind_speed_mph']} mph "
                  f"@ {weather['wind_dir_deg']}°  ({weather['condition']})")

        # Probable starters
        home_sp_name, away_sp_name = self._get_probable_starters(game_id)
        print(f"    SP: {away_sp_name} (away) vs {home_sp_name} (home)")

        # Bullpen HR/9 for each team
        bull_hr9 = self._bullpen_hr9(tpit_df)

        # Lineups
        away_lineup, home_lineup = self._get_lineups(game_id, game)

        rows = []

        # Away batters face home team's starter
        for batter in away_lineup:
            row = self._build_row(
                batter=batter,
                bat_df=bat_df,
                pit_df=pit_df,
                starter_name=home_sp_name,
                bullpen_hr9=bull_hr9.get(home_abbr, 1.3),
                park_team=home_abbr,
                weather=weather,
                dome=dome,
                model=model,
                game_info={
                    "game_id":    game_id,
                    "opponent":   home_team,
                    "at_venue":   venue,
                    "team":       away_team,
                    "team_abbr":  away_abbr,
                    "opp_starter": home_sp_name,
                },
            )
            if row:
                rows.append(row)

        # Home batters face away team's starter
        for batter in home_lineup:
            row = self._build_row(
                batter=batter,
                bat_df=bat_df,
                pit_df=pit_df,
                starter_name=away_sp_name,
                bullpen_hr9=bull_hr9.get(away_abbr, 1.3),
                park_team=home_abbr,
                weather=weather,
                dome=dome,
                model=model,
                game_info={
                    "game_id":    game_id,
                    "opponent":   away_team,
                    "at_venue":   venue,
                    "team":       home_team,
                    "team_abbr":  home_abbr,
                    "opp_starter": away_sp_name,
                },
            )
            if row:
                rows.append(row)

        return rows

    def _build_row(
        self, batter, bat_df, pit_df, starter_name,
        bullpen_hr9, park_team, weather, dome, model, game_info
    ):
        feat_vec = self.builder.build_prediction_row(
            batter_name=batter["name"],
            batter_mlbam_id=batter.get("id", 0),
            bat_df=bat_df,
            pit_df=pit_df,
            pit_name=starter_name,
            bullpen_hr9=bullpen_hr9,
            park_team=park_team,
            weather=weather or {},
            is_dome=dome,
        )
        if feat_vec is None:
            return None

        prob = float(model.predict_proba_hr(feat_vec.reshape(1, -1))[0])

        park  = get_park(park_team)
        winfo = weather or {}

        from src.ballparks import wind_factor as wf
        wfactor = 0.0
        if not dome and winfo:
            wfactor = wf(
                winfo.get("wind_speed_mph", 0),
                winfo.get("wind_dir_deg", 0),
                park["cf_bearing"],
            )

        return {
            "player_name":     batter["name"],
            "team":            game_info["team"],
            "team_abbr":       game_info["team_abbr"],
            "opponent":        game_info["opponent"],
            "opp_starter":     game_info["opp_starter"],
            "venue":           game_info["at_venue"],
            "park_hr_factor":  park["hr_factor"],
            "weather_temp":    winfo.get("temp_f", "N/A"),
            "weather_wind":    f"{winfo.get('wind_speed_mph','?')} mph",
            "weather_dir":     winfo.get("wind_dir_deg", 0),
            "weather_cond":    winfo.get("condition", "Dome"),
            "wind_factor":     wfactor,
            "hr_probability":  round(prob, 4),
            "hr_pct_display":  f"{prob * 100:.1f}%",
            "is_dome":         dome,
        }

    # ════════════════════════════════════════════════════════════════════
    # STATS API HELPERS
    # ════════════════════════════════════════════════════════════════════

    def _get_todays_games(self) -> list[dict]:
        try:
            sched = statsapi.schedule(date=self.today, sportId=1)
        except Exception as exc:
            print(f"  [statsapi] Error fetching schedule: {exc}")
            return []

        games = []
        for g in sched:
            if g.get("status") in ("Final", "Cancelled", "Postponed"):
                continue
            games.append({
                "game_id":       g["game_id"],
                "home_name":     g.get("home_name", ""),
                "away_name":     g.get("away_name", ""),
                "home_abbr":     self._abbr_from_name(g.get("home_name", "")),
                "away_abbr":     self._abbr_from_name(g.get("away_name", "")),
                "venue_name":    g.get("venue_name", ""),
                "game_datetime": g.get("game_datetime", ""),
                "status":        g.get("status", ""),
            })
        return games

    def _get_probable_starters(self, game_id: int) -> tuple[str, str]:
        try:
            data    = statsapi.boxscore_data(game_id)
            home_sp = data.get("home", {}).get("probablePitcher", {}).get("fullName", "Unknown")
            away_sp = data.get("away", {}).get("probablePitcher", {}).get("fullName", "Unknown")
            return home_sp, away_sp
        except Exception:
            pass

        try:
            game_info = statsapi.game(game_id)
            gd = game_info.get("gameData", {})
            probs = gd.get("probablePitchers", {})
            home_sp = probs.get("home", {}).get("fullName", "Unknown")
            away_sp = probs.get("away", {}).get("fullName", "Unknown")
            return home_sp, away_sp
        except Exception as exc:
            print(f"    [warn] Could not get probable starters: {exc}")
            return "Unknown", "Unknown"

    def _get_lineups(self, game_id: int, game: dict) -> tuple[list, list]:
        """
        Return (away_batters, home_batters) as lists of {name, id} dicts.
        Falls back to recent roster if lineup not posted yet.
        """
        try:
            box = statsapi.boxscore_data(game_id)

            def parse_order(side_data: dict) -> list:
                players = side_data.get("players", {})
                ordered = []
                for pid, pdata in players.items():
                    pos = pdata.get("position", {}).get("abbreviation", "")
                    batting_order = pdata.get("battingOrder")
                    if batting_order and pos != "P":
                        ordered.append({
                            "id":   pdata.get("person", {}).get("id"),
                            "name": pdata.get("person", {}).get("fullName", ""),
                            "batting_order": int(str(batting_order)[:2]),
                        })
                ordered.sort(key=lambda x: x["batting_order"])
                return ordered

            away = parse_order(box.get("away", {}))
            home = parse_order(box.get("home", {}))

            if away and home:
                return away, home
        except Exception:
            pass

        # Fallback: use 25-man roster hitters
        away = self._roster_hitters(game["away_abbr"])
        home = self._roster_hitters(game["home_abbr"])
        return away, home

    def _roster_hitters(self, team_abbr: str) -> list:
        """Get active position players from team roster."""
        try:
            team_id = statsapi.lookup_team(team_abbr)
            if not team_id:
                return []
            tid = team_id[0]["id"]
            roster = statsapi.roster(tid, rosterType="active")
            hitters = []
            for line in roster.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Line format: "#NN  POS  Full Name"
                parts = line.split()
                if len(parts) >= 3 and parts[1] not in ("P", "SP", "RP"):
                    name = " ".join(parts[2:])
                    hitters.append({"id": None, "name": name})
            return hitters[:9]       # Return top 9 as rough lineup estimate
        except Exception as exc:
            print(f"    [warn] Could not get roster for {team_abbr}: {exc}")
            return []

    def _bullpen_hr9(self, tpit_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute team HR/9 per team from team pitching stats.
        Returns dict keyed by team ABBREVIATION (e.g. 'NYY', 'LAD').

        FanGraphs team_pitching uses full team names (e.g. 'Yankees'), so we
        map those to abbreviations here so that downstream lookups of the form
        bull_hr9.get(home_abbr) and bull_hr9.get(away_abbr) work correctly.
        """
        if tpit_df.empty:
            return {}

        hr9_col = next(
            (c for c in ["HR/9", "HR9", "HR_9", "HRper9"] if c in tpit_df.columns),
            None,
        )
        team_col = next(
            (c for c in ["Team", "team", "teamName"] if c in tpit_df.columns),
            None,
        )
        if hr9_col is None or team_col is None:
            return {}

        # FanGraphs team names -> standard abbreviations
        FG_TEAM_TO_ABBR = {
            "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK",
            "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
            "Cardinals": "STL", "Cubs": "CHC", "Diamondbacks": "ARI",
            "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
            "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM",
            "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD",
            "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX",
            "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
            "Rockies": "COL", "Royals": "KC", "Tigers": "DET",
            "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY",
        }

        result = {}
        for _, row in tpit_df.iterrows():
            fg_team_name = str(row[team_col]).strip()
            abbr = FG_TEAM_TO_ABBR.get(fg_team_name)
            if abbr is None:
                # Last-resort: try the raw value as-is (already an abbr)
                abbr = fg_team_name
            val = pd.to_numeric(row[hr9_col], errors="coerce")
            if not pd.isna(val):
                result[abbr] = float(val)
        return result

    @staticmethod
    def _abbr_from_name(name: str) -> str:
        """Map full team name → abbreviation.  Very rough heuristic."""
        MAPPING = {
            "Orioles": "BAL", "Red Sox": "BOS", "Yankees": "NYY",
            "Rays": "TB", "Blue Jays": "TOR",
            "White Sox": "CWS", "Guardians": "CLE", "Tigers": "DET",
            "Royals": "KC", "Twins": "MIN",
            "Astros": "HOU", "Angels": "LAA", "Athletics": "OAK",
            "Mariners": "SEA", "Rangers": "TEX",
            "Braves": "ATL", "Marlins": "MIA", "Mets": "NYM",
            "Phillies": "PHI", "Nationals": "WSH",
            "Cubs": "CHC", "Reds": "CIN", "Brewers": "MIL",
            "Pirates": "PIT", "Cardinals": "STL",
            "Diamondbacks": "ARI", "Rockies": "COL", "Dodgers": "LAD",
            "Padres": "SD", "Giants": "SF",
        }
        for keyword, abbr in MAPPING.items():
            if keyword.lower() in name.lower():
                return abbr
        return name[:3].upper()
