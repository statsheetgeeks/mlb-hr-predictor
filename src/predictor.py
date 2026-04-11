"""
src/predictor.py
Fetches today's MLB schedule, lineups, and probable starters via the MLB
Stats API (statsapi package), then runs the trained Random Forest to rank
hitters by their home-run probability.
"""

from datetime import date
import numpy as np
import pandas as pd
import statsapi

from src.ballparks import get_park, is_dome
from src.weather import get_weather
from src.features import FeatureBuilder, _league_avg_pitcher
from src.model import HRModel
from src.data_fetcher import DataFetcher
from config import CURRENT_YEAR, MIN_PA_CURRENT_SEASON, TOP_N


# ── Hardcoded MLB team abbreviation → Stats API team ID ──────────────────────
# These IDs are permanent and never change.
TEAM_ID_MAP = {
    "BAL": 110, "BOS": 111, "NYY": 147, "TB":  139, "TOR": 141,
    "CWS": 145, "CLE": 114, "DET": 116, "KC":  118, "MIN": 142,
    "HOU": 117, "LAA": 108, "OAK": 133, "SEA": 136, "TEX": 140,
    "ATL": 144, "MIA": 146, "NYM": 121, "PHI": 143, "WSH": 120,
    "CHC": 112, "CIN": 113, "MIL": 158, "PIT": 134, "STL": 138,
    "ARI": 109, "COL": 115, "LAD": 119, "SD":  135, "SF":  137,
}


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
        Main entry point. Returns a list of prediction dicts sorted by
        hr_probability descending.
        """
        print(f"\nFetching today's schedule ({self.today})...")
        games = self._get_todays_games()
        if not games:
            print("No MLB games scheduled today.")
            return []

        print(f"Found {len(games)} games.")

        # Load current-season stats; fall back to Statcast-derived if blocked
        bat_df = self.fetcher.get_batting_stats(CURRENT_YEAR)
        if bat_df.empty:
            print("  [fallback] Deriving batting stats from Statcast...")
            sc_curr = self.fetcher.get_statcast_season(CURRENT_YEAR)
            bat_df  = self.fetcher.derive_batting_stats(sc_curr)

        pit_df = self.fetcher.get_pitching_stats(CURRENT_YEAR)
        if pit_df.empty:
            print("  [fallback] Deriving pitching stats from Statcast...")
            sc_curr = self.fetcher.get_statcast_season(CURRENT_YEAR)
            pit_df  = self.fetcher.derive_pitching_stats(sc_curr)

        tpit_df = self.fetcher.get_team_pitching(CURRENT_YEAR)

        print(f"  Batting stats: {len(bat_df)} players  |  "
              f"Pitching stats: {len(pit_df)} players")

        predictions = []
        for game in games:
            try:
                rows = self._predict_game(game, model, bat_df, pit_df, tpit_df)
                predictions.extend(rows)
            except Exception as exc:
                print(f"  [error] game {game.get('game_id')}: {exc}")

        print(f"\nTotal prediction rows: {len(predictions)}")

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

        game_id   = game["game_id"]
        home_team = game["home_name"]
        away_team = game["away_name"]
        home_abbr = game.get("home_abbr", "")
        away_abbr = game.get("away_abbr", "")
        venue     = game.get("venue_name", home_team)

        # Probable pitchers embedded from the schedule hydration call
        home_sp_name = game.get("home_probable_pitcher", "Unknown")
        away_sp_name = game.get("away_probable_pitcher", "Unknown")

        print(f"\n  {away_team} @ {home_team}")
        print(f"    SP: {away_sp_name} (away) vs {home_sp_name} (home)")

        # Park & weather
        park    = get_park(home_abbr)
        dome    = is_dome(home_abbr)
        weather = get_weather(park["lat"], park["lon"]) if not dome else None

        if weather:
            print(f"    Weather: {weather['temp_f']}F  "
                  f"Wind {weather['wind_speed_mph']} mph  "
                  f"{weather['condition']}")

        # Bullpen HR/9 for each team
        bull_hr9 = self._bullpen_hr9(tpit_df)

        # Lineups
        away_lineup, home_lineup = self._get_lineups(game_id, game)
        print(f"    Lineups: {len(away_lineup)} away, {len(home_lineup)} home batters")

        rows = []

        # Away batters face the HOME team's starter and HOME team's bullpen
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
                    "game_id":     game_id,
                    "opponent":    home_team,
                    "at_venue":    venue,
                    "team":        away_team,
                    "team_abbr":   away_abbr,
                    "opp_starter": home_sp_name,
                },
            )
            if row:
                rows.append(row)

        # Home batters face the AWAY team's starter and AWAY team's bullpen
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
                    "game_id":     game_id,
                    "opponent":    away_team,
                    "at_venue":    venue,
                    "team":        home_team,
                    "team_abbr":   home_abbr,
                    "opp_starter": away_sp_name,
                },
            )
            if row:
                rows.append(row)

        print(f"    Prediction rows: {len(rows)}")
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
            "player_name":    batter["name"],
            "team":           game_info["team"],
            "team_abbr":      game_info["team_abbr"],
            "opponent":       game_info["opponent"],
            "opp_starter":    game_info["opp_starter"],
            "venue":          game_info["at_venue"],
            "park_hr_factor": park["hr_factor"],
            "weather_temp":   winfo.get("temp_f", "N/A"),
            "weather_wind":   f"{winfo.get('wind_speed_mph','?')} mph",
            "weather_dir":    winfo.get("wind_dir_deg", 0),
            "weather_cond":   winfo.get("condition", "Dome"),
            "wind_factor":    wfactor,
            "hr_probability": round(prob, 4),
            "hr_pct_display": f"{prob * 100:.1f}%",
            "is_dome":        dome,
        }

    # ════════════════════════════════════════════════════════════════════
    # SCHEDULE  (raw API with probable pitcher hydration)
    # ════════════════════════════════════════════════════════════════════

    def _get_todays_games(self) -> list[dict]:
        """
        Fetch today's schedule using the raw statsapi.get() call with
        probablePitcher hydration so we get starter names in one request
        without needing separate per-game API calls.
        """
        try:
            raw = statsapi.get("schedule", {
                "sportId": 1,
                "date":    self.today,
                "hydrate": "probablePitcher",
            })
        except Exception as exc:
            print(f"  [statsapi] Schedule fetch error: {exc}")
            return []

        dates = raw.get("dates", [])
        if not dates:
            print(f"  [statsapi] No games found for {self.today}.")
            return []

        SKIP_STATUSES = {
            "Final", "Game Over", "Completed Early",
            "Postponed", "Cancelled", "Suspended", "Forfeit",
        }

        games = []
        for date_entry in dates:
            for g in date_entry.get("games", []):
                status_obj = g.get("status", {})
                abstract   = status_obj.get("abstractGameState", "")
                detailed   = status_obj.get("detailedState", "")

                if abstract == "Final" or detailed in SKIP_STATUSES:
                    continue

                away_data = g.get("teams", {}).get("away", {})
                home_data = g.get("teams", {}).get("home", {})

                away_name = away_data.get("team", {}).get("name", "")
                home_name = home_data.get("team", {}).get("name", "")

                # Probable pitchers come from the hydrated schedule response
                away_sp = (away_data.get("probablePitcher") or {}).get("fullName", "Unknown")
                home_sp = (home_data.get("probablePitcher") or {}).get("fullName", "Unknown")

                home_abbr = self._abbr_from_name(home_name)
                away_abbr = self._abbr_from_name(away_name)

                game_id = g.get("gamePk")
                print(f"    {away_name} @ {home_name}  [{detailed}]  "
                      f"SP: {away_sp} vs {home_sp}")

                games.append({
                    "game_id":               game_id,
                    "home_name":             home_name,
                    "away_name":             away_name,
                    "home_abbr":             home_abbr,
                    "away_abbr":             away_abbr,
                    "venue_name":            g.get("venue", {}).get("name", ""),
                    "game_datetime":         g.get("gameDate", ""),
                    "status":                detailed,
                    "home_probable_pitcher": home_sp,
                    "away_probable_pitcher": away_sp,
                })

        return games

    # ════════════════════════════════════════════════════════════════════
    # LINEUPS  (boxscore batting order -> JSON roster fallback)
    # ════════════════════════════════════════════════════════════════════

    def _get_lineups(self, game_id: int, game: dict) -> tuple[list, list]:
        """
        Return (away_batters, home_batters).
        Tries the live boxscore batting order first, then falls back to
        the active JSON roster for each team.
        """
        # Attempt 1: official batting order from boxscore
        try:
            box = statsapi.boxscore_data(game_id)

            def parse_order(side_data: dict) -> list:
                players = side_data.get("players", {})
                ordered = []
                for pid, pdata in players.items():
                    pos           = pdata.get("position", {}).get("abbreviation", "")
                    batting_order = pdata.get("battingOrder")
                    if batting_order and pos != "P":
                        ordered.append({
                            "id":            pdata.get("person", {}).get("id"),
                            "name":          pdata.get("person", {}).get("fullName", ""),
                            "batting_order": int(str(batting_order)[:2]),
                        })
                ordered.sort(key=lambda x: x["batting_order"])
                return ordered

            away = parse_order(box.get("away", {}))
            home = parse_order(box.get("home", {}))
            if away and home:
                print(f"    Using confirmed lineup from boxscore.")
                return away, home
        except Exception:
            pass

        # Attempt 2: active roster via raw JSON endpoint
        away = self._roster_hitters(game["away_abbr"])
        home = self._roster_hitters(game["home_abbr"])
        return away, home

    def _roster_hitters(self, team_abbr: str) -> list:
        """
        Fetch active position players via the raw JSON roster endpoint.
        Uses hardcoded team IDs — avoids the broken lookup_team(abbr) approach.
        """
        team_id = TEAM_ID_MAP.get(team_abbr.upper())
        if team_id is None:
            print(f"    [warn] No team ID for '{team_abbr}'")
            return []
        try:
            data = statsapi.get("teams_roster", {
                "teamId":     team_id,
                "rosterType": "active",
            })
            hitters = []
            for player in data.get("roster", []):
                pos  = player.get("position", {}).get("abbreviation", "")
                # Exclude pure pitchers; TWP (Two-Way Players) are included as batters
                if pos not in ("P", "SP", "RP"):
                    name = player.get("person", {}).get("fullName", "")
                    pid  = player.get("person", {}).get("id")
                    if name:
                        hitters.append({"id": pid, "name": name})
            print(f"    [roster] {team_abbr}: {len(hitters)} position players")
            return hitters
        except Exception as exc:
            print(f"    [warn] Roster error {team_abbr} (id={team_id}): {exc}")
            return []

    # ════════════════════════════════════════════════════════════════════
    # BULLPEN
    # ════════════════════════════════════════════════════════════════════

    def _bullpen_hr9(self, tpit_df: pd.DataFrame) -> dict[str, float]:
        """Return team HR/9 keyed by team abbreviation."""
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
            fg_name = str(row[team_col]).strip()
            abbr    = FG_TEAM_TO_ABBR.get(fg_name, fg_name)
            val     = pd.to_numeric(row[hr9_col], errors="coerce")
            if not pd.isna(val):
                result[abbr] = float(val)
        return result

    # ════════════════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def _abbr_from_name(name: str) -> str:
        """Map full MLB team name to standard abbreviation."""
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
