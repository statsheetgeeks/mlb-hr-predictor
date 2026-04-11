"""
src/data_fetcher.py
Handles all pybaseball data retrieval with intelligent caching.

Cache strategy
--------------
  Past seasons  → fetched once, stored as parquet/CSV, never re-fetched.
  Current season → stored with a "last_date" metadata file; only new dates
                   are downloaded each time the script runs.
"""

import os
import json
import time
from datetime import date, datetime, timedelta

import pandas as pd
import pybaseball

from config import (
    CACHE_DIR, CURRENT_YEAR, TRAINING_YEARS, ALL_YEARS,
    SEASON_START_MMDD, SEASON_END_MMDD,
)

# ── Silence pybaseball progress bars in CI environments ──────────────────────
pybaseball.cache.enable()


class DataFetcher:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def get_statcast_all_years(self) -> dict[int, pd.DataFrame]:
        """Return {year: statcast_df} for ALL_YEARS."""
        return {yr: self.get_statcast_season(yr) for yr in ALL_YEARS}

    def get_batting_stats_all_years(self) -> dict[int, pd.DataFrame]:
        """Return {year: batting_stats_df} for ALL_YEARS."""
        return {yr: self.get_batting_stats(yr) for yr in ALL_YEARS}

    def get_pitching_stats_all_years(self) -> dict[int, pd.DataFrame]:
        """Return {year: pitching_stats_df} for ALL_YEARS."""
        return {yr: self.get_pitching_stats(yr) for yr in ALL_YEARS}

    # ═══════════════════════════════════════════════════════════════════════
    # STATCAST  (pitch-level; we aggregate to game level in features.py)
    # ═══════════════════════════════════════════════════════════════════════

    def get_statcast_season(self, year: int) -> pd.DataFrame:
        cache_file = os.path.join(CACHE_DIR, f"statcast_{year}.parquet")
        meta_file  = os.path.join(CACHE_DIR, f"statcast_{year}_meta.json")

        if year < CURRENT_YEAR:
            # ── Past season: fetch once ───────────────────────────────────
            if os.path.exists(cache_file):
                print(f"  [cache] Loading statcast {year} from cache.")
                return pd.read_parquet(cache_file)

            print(f"  [fetch] Downloading full statcast {year} (may take several minutes)…")
            df = self._fetch_statcast_by_month(year)
            self._save_parquet(df, cache_file)
            return df

        else:
            # ── Current season: incremental update ───────────────────────
            return self._update_current_season_statcast(
                year, cache_file, meta_file
            )

    def _fetch_statcast_by_month(self, year: int) -> pd.DataFrame:
        """Fetch a full season in monthly chunks to avoid timeouts."""
        chunks = self._season_month_ranges(year)
        dfs = []
        for start_dt, end_dt in chunks:
            print(f"    fetching {start_dt} → {end_dt}")
            try:
                df = pybaseball.statcast(
                    start_dt=start_dt, end_dt=end_dt, parallel=True
                )
                if df is not None and len(df) > 0:
                    dfs.append(df)
            except Exception as exc:
                print(f"    WARNING: {exc}")
            time.sleep(2)          # be polite to the server

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _update_current_season_statcast(
        self, year: int, cache_file: str, meta_file: str
    ) -> pd.DataFrame:
        today     = date.today()
        yesterday = today - timedelta(days=1)   # completed games only

        # Load existing cache
        if os.path.exists(cache_file):
            existing = pd.read_parquet(cache_file)
            meta     = self._load_meta(meta_file)
            last_dt  = meta.get("last_date")
            if last_dt:
                last_date = datetime.strptime(last_dt, "%Y-%m-%d").date()
            else:
                last_date = pd.to_datetime(
                    existing["game_date"]
                ).dt.date.max() if len(existing) > 0 else None
        else:
            existing  = pd.DataFrame()
            last_date = None

        fetch_start = (
            last_date + timedelta(days=1)
            if last_date
            else date(year, 3, 20)
        )

        if fetch_start > yesterday:
            print(f"  [cache] Statcast {year} is up-to-date (last: {last_date}).")
            return existing

        print(
            f"  [fetch] Updating statcast {year}: "
            f"{fetch_start.isoformat()} → {yesterday.isoformat()}"
        )

        new_dfs = []
        for start_dt, end_dt in self._date_chunks(fetch_start, yesterday, days=7):
            try:
                df = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt)
                if df is not None and len(df) > 0:
                    new_dfs.append(df)
                else:
                    # No data returned → season hasn't started yet for this range
                    print(f"    No data for {start_dt}–{end_dt}; stopping search.")
                    break
            except Exception as exc:
                print(f"    WARNING fetching {start_dt}: {exc}")
                break
            time.sleep(1)

        if new_dfs:
            combined = pd.concat(
                [existing] + new_dfs, ignore_index=True
            ).drop_duplicates()
            self._save_parquet(combined, cache_file)
            new_last = pd.to_datetime(combined["game_date"]).dt.date.max()
            self._save_meta(meta_file, {"last_date": new_last.isoformat()})
            return combined

        return existing

    # ═══════════════════════════════════════════════════════════════════════
    # BATTING STATS  (FanGraphs season-level via pybaseball)
    # ═══════════════════════════════════════════════════════════════════════

    def get_batting_stats(self, year: int) -> pd.DataFrame:
        cache_file = os.path.join(CACHE_DIR, f"batting_{year}.csv")

        # Refresh current-year stats every time (stats change daily)
        if year < CURRENT_YEAR and os.path.exists(cache_file):
            print(f"  [cache] Loading batting stats {year}.")
            return pd.read_csv(cache_file)

        print(f"  [fetch] Downloading batting stats {year}…")
        try:
            df = pybaseball.batting_stats(year, qual=1)
            df.to_csv(cache_file, index=False)
            return df
        except Exception as exc:
            print(f"  WARNING: Could not fetch batting stats {year}: {exc}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # PITCHING STATS  (FanGraphs season-level)
    # ═══════════════════════════════════════════════════════════════════════

    def get_pitching_stats(self, year: int) -> pd.DataFrame:
        cache_file = os.path.join(CACHE_DIR, f"pitching_{year}.csv")

        if year < CURRENT_YEAR and os.path.exists(cache_file):
            print(f"  [cache] Loading pitching stats {year}.")
            return pd.read_csv(cache_file)

        print(f"  [fetch] Downloading pitching stats {year}…")
        try:
            df = pybaseball.pitching_stats(year, qual=1)
            df.to_csv(cache_file, index=False)
            return df
        except Exception as exc:
            print(f"  WARNING: Could not fetch pitching stats {year}: {exc}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # TEAM PITCHING (for bullpen stats)
    # ═══════════════════════════════════════════════════════════════════════

    def get_team_pitching(self, year: int) -> pd.DataFrame:
        cache_file = os.path.join(CACHE_DIR, f"team_pitching_{year}.csv")

        if year < CURRENT_YEAR and os.path.exists(cache_file):
            return pd.read_csv(cache_file)

        print(f"  [fetch] Downloading team pitching stats {year}…")
        try:
            df = pybaseball.team_pitching(year)
            df.to_csv(cache_file, index=False)
            return df
        except Exception as exc:
            print(f"  WARNING: {exc}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _season_month_ranges(year: int) -> list[tuple[str, str]]:
        """Return (start, end) strings for each month of an MLB season."""
        ranges = []
        season_months = [
            (f"{year}-03-20", f"{year}-03-31"),
            (f"{year}-04-01", f"{year}-04-30"),
            (f"{year}-05-01", f"{year}-05-31"),
            (f"{year}-06-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-07-31"),
            (f"{year}-08-01", f"{year}-08-31"),
            (f"{year}-09-01", f"{year}-09-30"),
            (f"{year}-10-01", f"{year}-10-05"),
        ]
        today = date.today()
        for s, e in season_months:
            start_d = datetime.strptime(s, "%Y-%m-%d").date()
            if start_d <= today:
                ranges.append((s, e))
        return ranges

    @staticmethod
    def _date_chunks(
        start: date, end: date, days: int = 7
    ) -> list[tuple[str, str]]:
        """Split [start, end] into chunks of `days` days."""
        chunks = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=days - 1), end)
            chunks.append((cur.isoformat(), chunk_end.isoformat()))
            cur = chunk_end + timedelta(days=1)
        return chunks

    @staticmethod
    def _save_parquet(df: pd.DataFrame, path: str):
        if df is None or len(df) == 0:
            return
        # Coerce mixed-type columns to string to avoid arrow errors
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(str)
                except Exception:
                    pass
        df.to_parquet(path, index=False, engine="pyarrow")

    @staticmethod
    def _load_meta(path: str) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_meta(path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f)
