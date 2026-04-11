"""
src/features.py
Transforms raw pybaseball / statsapi data into the feature matrix used by
the Random Forest.

Training features (per batter-game row):
  – Batter:   season HR%, ISO, BB%, K%, barrel%, exit_velo, hard_hit%,
              launch_angle_mean, pull%
  – Pitcher:  HR/9, xFIP, barrel%_allowed, exit_velo_allowed, K/9, BB/9
  – Bullpen:  team HR/9 (starters excluded via FIP proxy)
  – Park:     HR park factor, elevation_ft (thin air → more HRs)
  – Weather:  temp_f, wind_factor, humidity_pct

Target: did the batter hit a home run in that game? (0 / 1)
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.ballparks import get_park, wind_factor


# ── Column name helpers ───────────────────────────────────────────────────────

# FanGraphs batting column names (as returned by pybaseball.batting_stats)
BAT_COLS = {
    "name":       "Name",
    "team":       "Team",
    "pa":         "PA",
    "hr":         "HR",
    "bb_pct":     "BB%",
    "k_pct":      "K%",
    "iso":        "ISO",
    "avg":        "AVG",
    "slg":        "SLG",
    "obp":        "OBP",
    "barrel_pct": "Barrel%",
    "exit_velo":  "EV",
    "hard_pct":   "Hard%",
    "la":         "LA",
    "pull_pct":   "Pull%",
    "hr_fb":      "HR/FB",
    "xslg":       "xSLG",
}

# FanGraphs pitching column names
PIT_COLS = {
    "name":       "Name",
    "team":       "Team",
    "ip":         "IP",
    "hr9":        "HR/9",
    "xfip":       "xFIP",
    "barrel_pct": "Barrel%",
    "exit_velo":  "EV",
    "k9":         "K/9",
    "bb9":        "BB/9",
    "hard_pct":   "Hard%",
    "era":        "ERA",
}

FEATURE_NAMES = [
    # Batter
    "bat_hr_rate",       # HR / PA  (season)
    "bat_iso",
    "bat_bb_pct",
    "bat_k_pct",
    "bat_barrel_pct",
    "bat_exit_velo",
    "bat_hard_pct",
    "bat_launch_angle",
    "bat_pull_pct",
    "bat_hr_fb",
    "bat_xslg",
    "bat_pa",            # proxy for sample size / confidence
    # Pitcher
    "pit_hr9",
    "pit_xfip",
    "pit_barrel_pct",
    "pit_exit_velo",
    "pit_k9",
    "pit_bb9",
    "pit_hard_pct",
    # Bullpen
    "bull_hr9",
    # Park
    "park_hr_factor",
    "park_elevation_ft",
    # Weather
    "weather_temp_f",
    "weather_wind_factor",
    "weather_humidity",
]


class FeatureBuilder:

    # ════════════════════════════════════════════════════════════════════
    # TRAINING  (historical statcast + season stats → X, y)
    # ════════════════════════════════════════════════════════════════════

    def build_training_features(
        self,
        statcast_by_year: dict,
        batting_by_year: dict,
        pitching_by_year: dict,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build X (feature matrix) and y (HR labels) from multiple seasons.

        For each (batter, game) in statcast:
          - label  = 1 if batter hit a HR in that game
          - features = batter's full-season stats + the stats of the first
            OPPOSING pitcher the batter faced that game (i.e. the starter).

        Batter-pitcher pairing is taken directly from Statcast matchup records.
        Statcast records only real in-game matchups, so a batter entry always
        corresponds to an opposing pitcher — a batter never faces a teammate.
        The first pitcher (by at_bat_number) a batter faces in a game is the
        opposing team's starting pitcher.

        Player names are used for all FanGraphs lookups because Statcast uses
        MLBAM IDs while FanGraphs uses its own separate numeric IDs. Names are
        the correct shared key between the two data sources.

        Returns: X (N x F), y (N,), feature_names (list)
        """
        all_rows = []

        for year, sc_df in statcast_by_year.items():
            if sc_df is None or len(sc_df) == 0:
                continue

            bat_df = batting_by_year.get(year, pd.DataFrame())
            pit_df = pitching_by_year.get(year, pd.DataFrame())

            if bat_df.empty or pit_df.empty:
                print(f"  Skipping {year}: missing batting/pitching stats.")
                continue

            print(f"  Building training rows for {year}...")

            sc_sorted = sc_df.sort_values("at_bat_number")

            # ── Game-level HR labels keyed by (game_date, batter MLBAM ID) ──
            sc_pa = sc_sorted[sc_sorted["events"].notna()].copy()
            sc_pa["is_hr"] = (sc_pa["events"] == "home_run").astype(int)
            game_labels = (
                sc_pa.groupby(["game_date", "batter"])["is_hr"]
                .max()       # 1 if the batter hit >= 1 HR that game, else 0
                .reset_index()
            )

            # ── First opposing pitcher faced per batter per game ────────────
            # Statcast records real matchups only — the pitcher column always
            # refers to the opposing pitcher, never a teammate. Taking the
            # first at_bat_number gives us the opposing starter.
            first_pitcher = (
                sc_sorted.groupby(["game_date", "batter"])["pitcher"]
                .first()
                .reset_index()
                .rename(columns={"pitcher": "opp_starter_mlbam"})
            )

            # ── Batter name from Statcast (player_name = batting player) ───
            # player_name is the batter; it is constant within a
            # (game_date, batter) group so first() is safe.
            batter_names = (
                sc_sorted.groupby(["game_date", "batter"])["player_name"]
                .first()
                .reset_index()
                .rename(columns={"player_name": "batter_name"})
            )

            # ── Resolve pitcher MLBAM IDs to names ─────────────────────────
            # We need names to join against FanGraphs pitching stats. We build
            # a batch MLBAM-ID -> name mapping for all pitchers in this season.
            unique_pitcher_ids = (
                first_pitcher["opp_starter_mlbam"]
                .dropna().astype(int).unique().tolist()
            )
            pitcher_id_to_name = _build_pitcher_name_map(unique_pitcher_ids)

            first_pitcher["opp_starter_name"] = (
                first_pitcher["opp_starter_mlbam"].map(pitcher_id_to_name)
            )

            # ── Merge into one game-info table ──────────────────────────────
            game_info = (
                game_labels
                .merge(batter_names,  on=["game_date", "batter"], how="left")
                .merge(first_pitcher, on=["game_date", "batter"], how="left")
            )

            # ── FanGraphs stat lookups keyed by player name ─────────────────
            # Names are used consistently here (training) and in predictions,
            # since FanGraphs and MLBAM use different numeric ID systems.
            bat_feats = self._batting_feature_dict_by_name(bat_df)
            pit_feats = self._pitching_feature_dict_by_name(pit_df)

            matched = 0
            for _, row in game_info.iterrows():
                batter_name  = str(row.get("batter_name",      "") or "").strip()
                pitcher_name = str(row.get("opp_starter_name", "") or "").strip()

                bfeat = bat_feats.get(batter_name)
                pfeat = pit_feats.get(pitcher_name)

                if bfeat is None:
                    continue   # batter not in FanGraphs (too few PA, etc.)
                if pfeat is None:
                    pfeat = _league_avg_pitcher()  # pitcher not in FanGraphs -> fallback

                feat_row = bfeat + pfeat + [
                    0.0,   # bull_hr9          – neutral for training
                    1.0,   # park_hr_factor    – neutral
                    600.0, # park_elevation_ft – average
                    72.0,  # weather_temp_f    – neutral
                    0.0,   # weather_wind_factor
                    50.0,  # weather_humidity
                ]
                all_rows.append((feat_row, int(row["is_hr"])))
                matched += 1

            print(f"    {year}: {matched:,} matched rows / {len(game_info):,} batter-game pairs")

        if not all_rows:
            raise RuntimeError("No training rows built – check data.")

        X = np.array([r[0] for r in all_rows], dtype=np.float32)
        y = np.array([r[1] for r in all_rows], dtype=np.int8)

        print(f"  Total training rows: {len(y):,}  |  HR rate: {y.mean():.4f}")
        return X, y, FEATURE_NAMES

    # ════════════════════════════════════════════════════════════════════
    # PREDICTION  (today's batters)
    # ════════════════════════════════════════════════════════════════════

    def build_prediction_row(
        self,
        batter_name: str,
        batter_mlbam_id: int,
        bat_df: pd.DataFrame,
        pit_df: pd.DataFrame,
        pit_name: str,
        bullpen_hr9: float,
        park_team: str,
        weather: dict,
        is_dome: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Build a single prediction feature vector for one batter today.
        Returns None if stats are unavailable.
        """
        bat_feats_dict = self._batting_feature_dict_by_name(bat_df)
        pit_feats_dict = self._pitching_feature_dict_by_name(pit_df)

        # Exact name lookup first; fall back to accent-stripped normalization
        bfeat = bat_feats_dict.get(batter_name)
        if bfeat is None:
            norm = _normalize_name(batter_name)
            bfeat = next(
                (v for k, v in bat_feats_dict.items() if _normalize_name(k) == norm),
                None,
            )

        pfeat = pit_feats_dict.get(pit_name)
        if pfeat is None and pit_name not in ("Unknown", ""):
            norm = _normalize_name(pit_name)
            pfeat = next(
                (v for k, v in pit_feats_dict.items() if _normalize_name(k) == norm),
                None,
            )

        if bfeat is None:
            return None
        if pfeat is None:
            # Use league-average pitcher profile
            pfeat = _league_avg_pitcher()

        park = get_park(park_team)
        wf   = 0.0
        if not is_dome and weather:
            wf = wind_factor(
                weather.get("wind_speed_mph", 0),
                weather.get("wind_dir_deg", 0),
                park["cf_bearing"],
            )

        feat_vec = bfeat + pfeat + [
            bullpen_hr9,
            park["hr_factor"],
            park["elevation_ft"],
            weather.get("temp_f", 72.0) if weather else 72.0,
            wf,
            weather.get("humidity_pct", 50.0) if weather else 50.0,
        ]

        return np.array(feat_vec, dtype=np.float32)

    # ════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ════════════════════════════════════════════════════════════════════

    def _batting_feature_dict_by_name(self, bat_df: pd.DataFrame) -> dict:
        mapping = {}
        feats = self._extract_batting_features(bat_df)
        for _, row in feats.iterrows():
            name = str(row.get("Name", "")).strip()
            if name:
                mapping[name] = row["__feat_list"]
        return mapping

    def _pitching_feature_dict_by_name(self, pit_df: pd.DataFrame) -> dict:
        mapping = {}
        feats = self._extract_pitching_features(pit_df)
        for _, row in feats.iterrows():
            name = str(row.get("Name", "")).strip()
            if name:
                mapping[name] = row["__feat_list"]
        return mapping

    def _extract_batting_features(self, bat_df: pd.DataFrame) -> pd.DataFrame:
        df = bat_df.copy()
        cols = {v: k for k, v in BAT_COLS.items()}

        def g(col_name, default=0.0):
            fg_name = BAT_COLS.get(col_name, col_name)
            if fg_name in df.columns:
                return pd.to_numeric(df[fg_name], errors="coerce").fillna(default)
            return pd.Series([default] * len(df))

        pa  = g("pa", 1.0).clip(lower=1)
        hr  = g("hr", 0.0)

        df["__hr_rate"]   = (hr / pa).clip(0, 1)
        df["__iso"]       = g("iso")
        df["__bb_pct"]    = _pct(g("bb_pct"))
        df["__k_pct"]     = _pct(g("k_pct"))
        df["__barrel"]    = _pct(g("barrel_pct"))
        df["__exit_velo"] = g("exit_velo", 88.0)
        df["__hard_pct"]  = _pct(g("hard_pct"))
        df["__la"]        = g("la", 12.0)
        df["__pull_pct"]  = _pct(g("pull_pct"))
        df["__hr_fb"]     = _pct(g("hr_fb"))
        df["__xslg"]      = g("xslg", 0.0)
        df["__pa"]        = pa

        df["__feat_list"] = df.apply(lambda r: [
            r["__hr_rate"], r["__iso"], r["__bb_pct"], r["__k_pct"],
            r["__barrel"], r["__exit_velo"], r["__hard_pct"], r["__la"],
            r["__pull_pct"], r["__hr_fb"], r["__xslg"], r["__pa"],
        ], axis=1)

        return df

    def _extract_pitching_features(self, pit_df: pd.DataFrame) -> pd.DataFrame:
        df = pit_df.copy()

        def g(col_name, default=0.0):
            fg_name = PIT_COLS.get(col_name, col_name)
            if fg_name in df.columns:
                return pd.to_numeric(df[fg_name], errors="coerce").fillna(default)
            return pd.Series([default] * len(df))

        df["__hr9"]       = g("hr9",  1.3)
        df["__xfip"]      = g("xfip", 4.0)
        df["__barrel"]    = _pct(g("barrel_pct"))
        df["__exit_velo"] = g("exit_velo", 88.0)
        df["__k9"]        = g("k9",  8.5)
        df["__bb9"]       = g("bb9",  3.0)
        df["__hard_pct"]  = _pct(g("hard_pct"))

        df["__feat_list"] = df.apply(lambda r: [
            r["__hr9"], r["__xfip"], r["__barrel"],
            r["__exit_velo"], r["__k9"], r["__bb9"], r["__hard_pct"],
        ], axis=1)

        return df


# ── Module-level utilities ────────────────────────────────────────────────────

def _pct(series: pd.Series) -> pd.Series:
    """Convert '12.3%' strings or already-numeric values to float [0-1]."""
    def convert(v):
        if isinstance(v, str):
            return float(v.strip("%")) / 100.0
        if v > 1.0:       # e.g. 12.3 instead of 0.123
            return v / 100.0
        return v
    return series.apply(convert)


def _normalize_name(name: str) -> str:
    """
    Strip accents and reduce to lowercase ASCII for fuzzy name matching.
    Handles mismatches like 'Jose Ramirez' vs 'José Ramírez', or
    'Vladimir Guerrero Jr.' vs 'Vladimir Guerrero Jr'.
    """
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    # Strip trailing punctuation differences (Jr., Sr., etc.)
    return ascii_str.strip().rstrip(".").strip().lower()


def _league_avg_pitcher() -> list:
    """League-average pitcher feature vector (fallback when starter unknown)."""
    return [1.3, 4.00, 0.076, 88.5, 8.5, 3.0, 0.34]


def _build_pitcher_name_map(mlbam_ids: list[int]) -> dict[int, str]:
    """
    Return a dict mapping MLBAM pitcher ID -> full name.

    Uses pybaseball.playerid_reverse_lookup() in a single batch call so we
    don't hammer the API for each pitcher individually. Falls back gracefully
    if the lookup fails or an ID is not found.

    The returned names match FanGraphs 'Name' column format so they can be
    used directly to look up pitcher features in _pitching_feature_dict_by_name.
    """
    if not mlbam_ids:
        return {}

    try:
        import pybaseball
        lookup_df = pybaseball.playerid_reverse_lookup(
            mlbam_ids, key_type="mlbam"
        )
        if lookup_df is None or lookup_df.empty:
            return {}

        # pybaseball returns: key_mlbam, name_first, name_last, ...
        result = {}
        for _, row in lookup_df.iterrows():
            mlbam_id   = int(row.get("key_mlbam", 0))
            first_name = str(row.get("name_first", "")).strip().title()
            last_name  = str(row.get("name_last",  "")).strip().title()
            if mlbam_id and first_name and last_name:
                # FanGraphs uses "First Last" format
                result[mlbam_id] = f"{first_name} {last_name}"
        return result

    except Exception as exc:
        print(f"  [warn] pitcher name lookup failed: {exc}")
        return {}
