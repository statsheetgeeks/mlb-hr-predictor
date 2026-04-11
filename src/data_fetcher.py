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
    # BATTING STATS  (FanGraphs → Baseball Reference → Statcast fallback)
    # ═══════════════════════════════════════════════════════════════════════

    def get_batting_stats(self, year: int) -> pd.DataFrame:
        """
        Fetch batter season stats with a three-tier fallback:
          1. FanGraphs via pybaseball.batting_stats()      – richest (Statcast metrics)
          2. Baseball Reference via pybaseball.batting_stats_bref()  – reliable fallback
          3. Empty DataFrame  (caller will derive from Statcast)
        All sources are normalized to FanGraphs column names so features.py
        works without modification regardless of which source is used.
        """
        cache_file = os.path.join(CACHE_DIR, f"batting_{year}.csv")

        # Past seasons: load from cache if available
        if year < CURRENT_YEAR and os.path.exists(cache_file):
            print(f"  [cache] Loading batting stats {year}.")
            return pd.read_csv(cache_file)

        # ── Tier 1: FanGraphs ─────────────────────────────────────────────
        print(f"  [fetch] Batting stats {year} – trying FanGraphs…")
        try:
            df = pybaseball.batting_stats(year, qual=1)
            if df is not None and not df.empty:
                df.to_csv(cache_file, index=False)
                print(f"    ✓ FanGraphs batting stats {year} ({len(df)} players)")
                return df
        except Exception as exc:
            print(f"    FanGraphs failed: {exc}")

        # ── Tier 2: Baseball Reference ────────────────────────────────────
        print(f"  [fetch] Batting stats {year} – trying Baseball Reference…")
        try:
            df = pybaseball.batting_stats_bref(year)
            if df is not None and not df.empty:
                df = _normalize_batting_bref(df)
                df.to_csv(cache_file, index=False)
                print(f"    ✓ BBRef batting stats {year} ({len(df)} players)")
                return df
        except Exception as exc:
            print(f"    Baseball Reference failed: {exc}")

        # ── Tier 3: signal to caller to derive from Statcast ─────────────
        print(f"  [warn] All batting stat sources failed for {year}.")
        return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # PITCHING STATS  (FanGraphs → Baseball Reference → Statcast fallback)
    # ═══════════════════════════════════════════════════════════════════════

    def get_pitching_stats(self, year: int) -> pd.DataFrame:
        """
        Fetch pitcher season stats with a three-tier fallback:
          1. FanGraphs via pybaseball.pitching_stats()
          2. Baseball Reference via pybaseball.pitching_stats_bref()
          3. Empty DataFrame  (caller will derive from Statcast)
        """
        cache_file = os.path.join(CACHE_DIR, f"pitching_{year}.csv")

        if year < CURRENT_YEAR and os.path.exists(cache_file):
            print(f"  [cache] Loading pitching stats {year}.")
            return pd.read_csv(cache_file)

        # ── Tier 1: FanGraphs ─────────────────────────────────────────────
        print(f"  [fetch] Pitching stats {year} – trying FanGraphs…")
        try:
            df = pybaseball.pitching_stats(year, qual=1)
            if df is not None and not df.empty:
                df.to_csv(cache_file, index=False)
                print(f"    ✓ FanGraphs pitching stats {year} ({len(df)} players)")
                return df
        except Exception as exc:
            print(f"    FanGraphs failed: {exc}")

        # ── Tier 2: Baseball Reference ────────────────────────────────────
        print(f"  [fetch] Pitching stats {year} – trying Baseball Reference…")
        try:
            df = pybaseball.pitching_stats_bref(year)
            if df is not None and not df.empty:
                df = _normalize_pitching_bref(df)
                df.to_csv(cache_file, index=False)
                print(f"    ✓ BBRef pitching stats {year} ({len(df)} players)")
                return df
        except Exception as exc:
            print(f"    Baseball Reference failed: {exc}")

        # ── Tier 3: signal to caller to derive from Statcast ─────────────
        print(f"  [warn] All pitching stat sources failed for {year}.")
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

    # ═══════════════════════════════════════════════════════════════════
    # DERIVED STATS  (fallback when FanGraphs is unavailable)
    # ═══════════════════════════════════════════════════════════════════

    def derive_batting_stats(self, sc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive batter season stats directly from Statcast pitch-level data.
        Used as a fallback when FanGraphs returns a 403 (blocks cloud IPs).
        Output columns match FanGraphs names so features.py works unchanged.
        """
        if sc_df is None or sc_df.empty:
            return pd.DataFrame()

        print("  [derive] Computing batting stats from Statcast data...")

        # PA-ending rows only
        pa_df = sc_df[sc_df["events"].notna()].copy()
        if pa_df.empty:
            return pd.DataFrame()

        # Batted-ball rows (type == 'X' in Statcast)
        bb_df = (
            sc_df[sc_df["type"] == "X"].copy()
            if "type" in sc_df.columns
            else pd.DataFrame()
        )

        # ── PA-level aggregation ───────────────────────────────────────────
        def _pa_agg(grp):
            pa      = len(grp)
            hr      = int((grp["events"] == "home_run").sum())
            bb      = int(grp["events"].isin(["walk", "intent_walk"]).sum())
            hbp     = int((grp["events"] == "hit_by_pitch").sum())
            k       = int(grp["events"].isin(["strikeout", "strikeout_double_play"]).sum())
            singles = int((grp["events"] == "single").sum())
            doubles = int((grp["events"] == "double").sum())
            triples = int((grp["events"] == "triple").sum())
            hits    = singles + doubles + triples + hr
            slg     = (singles + 2*doubles + 3*triples + 4*hr) / pa if pa else 0.0
            avg     = hits / pa if pa else 0.0
            obp     = (hits + bb + hbp) / pa if pa else 0.0
            return pd.Series({
                "PA":  pa,  "HR": hr,  "BB": bb,  "K": k,
                "HR_n": hr,           # kept for HR/FB calc
                "Hits": hits,
                "AVG":  avg, "SLG": slg, "ISO": slg - avg, "OBP": obp,
                "BB%":  (bb / pa * 100) if pa else 0.0,
                "K%":   (k  / pa * 100) if pa else 0.0,
            })

        pa_stats = (
            pa_df.groupby("player_name", group_keys=False)
            .apply(_pa_agg)
            .reset_index()
        )

        # ── Batted-ball aggregation ────────────────────────────────────────
        if not bb_df.empty and "player_name" in bb_df.columns:
            def _bb_agg(grp):
                n    = len(grp)
                ev   = pd.to_numeric(grp.get("launch_speed",   pd.Series(dtype=float)), errors="coerce")
                la   = pd.to_numeric(grp.get("launch_angle",   pd.Series(dtype=float)), errors="coerce")
                hard = float((ev >= 95).sum() / n) if n else 0.0
                brrl = (
                    pd.to_numeric(grp["barrel"], errors="coerce").mean()
                    if "barrel" in grp.columns else 0.0
                )
                n_fb = int((la >= 25).sum())
                return pd.Series({
                    "EV":      ev.mean()  if not ev.isna().all()  else 88.0,
                    "LA":      la.mean()  if not la.isna().all()  else 12.0,
                    "Hard%":   hard * 100,
                    "Barrel%": (brrl * 100) if not pd.isna(brrl) else 0.0,
                    "FB":      n_fb,
                })

            bb_stats = (
                bb_df.groupby("player_name", group_keys=False)
                .apply(_bb_agg)
                .reset_index()
            )
            pa_stats = pa_stats.merge(bb_stats, on="player_name", how="left")
            pa_stats["HR/FB"] = (
                pa_stats["HR_n"] / pa_stats["FB"].clip(lower=1) * 100
            ).fillna(0.0)
        else:
            pa_stats["EV"]      = 88.0
            pa_stats["LA"]      = 12.0
            pa_stats["Hard%"]   = 34.0
            pa_stats["Barrel%"] = 7.5
            pa_stats["HR/FB"]   = 0.0

        pa_stats["Pull%"] = 40.0                          # league-avg default
        pa_stats["xSLG"]  = pa_stats["SLG"].fillna(0.0)  # proxy

        pa_stats.rename(columns={"player_name": "Name"}, inplace=True)

        keep = [
            "Name", "PA", "HR", "BB%", "K%", "ISO",
            "AVG", "SLG", "OBP", "Barrel%", "EV",
            "Hard%", "LA", "Pull%", "HR/FB", "xSLG",
        ]
        return pa_stats[[c for c in keep if c in pa_stats.columns]]

    # ─────────────────────────────────────────────────────────────────────

    def derive_pitching_stats(self, sc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive pitcher season stats directly from Statcast data.
        Output columns match FanGraphs names so features.py works unchanged.
        """
        if sc_df is None or sc_df.empty:
            return pd.DataFrame()

        print("  [derive] Computing pitching stats from Statcast data...")

        pa_df = sc_df[sc_df["events"].notna()].copy()
        bb_df = (
            sc_df[sc_df["type"] == "X"].copy()
            if "type" in sc_df.columns
            else pd.DataFrame()
        )

        # Resolve pitcher MLBAM IDs → full names
        unique_ids = pa_df["pitcher"].dropna().astype(int).unique().tolist()
        id_to_name = self._resolve_pitcher_names(unique_ids)

        pa_df["pitcher_name"] = pa_df["pitcher"].map(id_to_name)
        pa_df = pa_df[pa_df["pitcher_name"].notna()]

        if pa_df.empty:
            return pd.DataFrame()

        if not bb_df.empty:
            bb_df["pitcher_name"] = bb_df["pitcher"].map(id_to_name)
            bb_df = bb_df[bb_df["pitcher_name"].notna()]

        # Events that record outs (for IP estimation)
        _SINGLE_OUTS = {
            "strikeout", "field_out", "force_out", "sac_fly",
            "sac_bunt", "fielders_choice_out", "fielders_choice",
        }
        _DOUBLE_OUTS = {
            "strikeout_double_play", "grounded_into_double_play",
            "double_play", "sac_fly_double_play",
        }
        _TRIPLE_OUTS = {"triple_play"}

        def _pit_agg(grp):
            pa  = len(grp)
            hr  = int((grp["events"] == "home_run").sum())
            bb  = int(grp["events"].isin(["walk", "intent_walk"]).sum())
            k   = int(grp["events"].isin(["strikeout", "strikeout_double_play"]).sum())
            outs = (
                int(grp["events"].isin(_SINGLE_OUTS).sum())
                + int(grp["events"].isin(_DOUBLE_OUTS).sum()) * 2
                + int(grp["events"].isin(_TRIPLE_OUTS).sum()) * 3
            )
            ip  = max(outs / 3.0, 1.0)      # avoid division by zero
            return pd.Series({
                "IP":    ip,
                "HR/9":  hr / ip * 9,
                "K/9":   k  / ip * 9,
                "BB/9":  bb / ip * 9,
                "xFIP":  4.00,               # league-avg placeholder
                "ERA":   4.00,
            })

        pit_stats = (
            pa_df.groupby("pitcher_name", group_keys=False)
            .apply(_pit_agg)
            .reset_index()
        )

        # Batted-ball stats allowed
        if not bb_df.empty and "pitcher_name" in bb_df.columns:
            def _pit_bb_agg(grp):
                n    = len(grp)
                ev   = pd.to_numeric(grp.get("launch_speed", pd.Series(dtype=float)), errors="coerce")
                hard = float((ev >= 95).sum() / n) if n else 0.0
                brrl = (
                    pd.to_numeric(grp["barrel"], errors="coerce").mean()
                    if "barrel" in grp.columns else 0.0
                )
                return pd.Series({
                    "EV":      ev.mean()  if not ev.isna().all()  else 88.0,
                    "Hard%":   hard * 100,
                    "Barrel%": (brrl * 100) if not pd.isna(brrl) else 0.0,
                })

            pit_bb = (
                bb_df.groupby("pitcher_name", group_keys=False)
                .apply(_pit_bb_agg)
                .reset_index()
            )
            pit_stats = pit_stats.merge(pit_bb, on="pitcher_name", how="left")
        else:
            pit_stats["EV"]      = 88.0
            pit_stats["Hard%"]   = 34.0
            pit_stats["Barrel%"] = 7.5

        pit_stats.rename(columns={"pitcher_name": "Name"}, inplace=True)

        keep = ["Name", "IP", "HR/9", "xFIP", "Barrel%", "EV", "K/9", "BB/9", "Hard%", "ERA"]
        return pit_stats[[c for c in keep if c in pit_stats.columns]]

    # ─────────────────────────────────────────────────────────────────────

    def _resolve_pitcher_names(self, mlbam_ids: list) -> dict:
        """
        Batch-resolve pitcher MLBAM IDs to full names via pybaseball.
        Results are cached locally so repeat calls are instant.
        """
        if not mlbam_ids:
            return {}

        cache_path = os.path.join(CACHE_DIR, "pitcher_id_name_cache.json")

        # Load existing cache
        cached: dict[int, str] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                cached = {}

        missing = [i for i in mlbam_ids if int(i) not in cached]

        if missing:
            try:
                import pybaseball as pyb
                lookup = pyb.playerid_reverse_lookup(missing, key_type="mlbam")
                if lookup is not None and not lookup.empty:
                    for _, row in lookup.iterrows():
                        mid   = int(row.get("key_mlbam", 0))
                        first = str(row.get("name_first", "")).strip().title()
                        last  = str(row.get("name_last",  "")).strip().title()
                        if mid and first and last:
                            cached[mid] = f"{first} {last}"
                with open(cache_path, "w") as f:
                    json.dump({str(k): v for k, v in cached.items()}, f)
            except Exception as exc:
                print(f"  [warn] pitcher name lookup failed: {exc}")

        return cached

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

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


# ── Module-level helpers ──────────────────────────────────────────────────────

def _normalize_batting_bref(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate Baseball Reference batting columns into FanGraphs naming so
    features.py works unchanged regardless of which source provided the data.

    BBRef has the core counting stats and rate stats but lacks Statcast
    metrics (barrel%, exit velocity, hard-hit%, xSLG).  Those are filled
    with league-average defaults so the feature vector stays complete.
    """
    def _num(col, default=0.0):
        return pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)

    out             = pd.DataFrame()
    out["Name"]     = df["Name"]
    out["PA"]       = _num("PA",  0.0).clip(lower=1)
    out["HR"]       = _num("HR",  0.0)

    bb              = _num("BB",  0.0)
    so              = _num("SO",  0.0)
    pa              = out["PA"]

    out["BB%"]      = bb / pa * 100          # stored as 0-100; _pct() converts
    out["K%"]       = so / pa * 100
    out["AVG"]      = _num("BA",  0.0)       # already a decimal (0.275)
    out["OBP"]      = _num("OBP", 0.0)
    out["SLG"]      = _num("SLG", 0.0)
    out["ISO"]      = out["SLG"] - out["AVG"]
    out["xSLG"]     = out["SLG"]             # SLG as proxy for xSLG

    # Statcast metrics absent from BBRef – league-average defaults
    out["Barrel%"]  = 7.5    # ~league avg barrel rate
    out["EV"]       = 88.0   # mph
    out["Hard%"]    = 34.0   # ~league avg hard-hit rate
    out["LA"]       = 12.0   # degrees
    out["Pull%"]    = 40.0   # %
    out["HR/FB"]    = 0.0    # unknown without Statcast; model will weight low

    return out[[
        "Name", "PA", "HR", "BB%", "K%", "ISO",
        "AVG", "SLG", "OBP", "Barrel%", "EV",
        "Hard%", "LA", "Pull%", "HR/FB", "xSLG",
    ]]


def _normalize_pitching_bref(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate Baseball Reference pitching columns into FanGraphs naming.

    BBRef pitching includes HR9, SO9, BB9, and FIP which map cleanly.
    Statcast metrics (barrel%, exit velocity, hard-hit%) are filled with
    league-average defaults.

    BBRef column names used:
      HR9  → HR/9    SO9 → K/9    BB9 → BB/9
      FIP  → xFIP    ERA → ERA    IP  → IP
    """
    def _num(col, default=0.0):
        return pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)

    out             = pd.DataFrame()
    out["Name"]     = df["Name"]
    out["IP"]       = _num("IP",   1.0).clip(lower=1)
    out["HR/9"]     = _num("HR9",  1.3)
    out["xFIP"]     = _num("FIP",  4.0)   # FIP is a good xFIP proxy
    out["ERA"]      = _num("ERA",  4.0)
    out["K/9"]      = _num("SO9",  8.5)
    out["BB/9"]     = _num("BB9",  3.0)

    # Statcast metrics absent from BBRef – league-average defaults
    out["Barrel%"]  = 7.5
    out["EV"]       = 88.0
    out["Hard%"]    = 34.0

    return out[[
        "Name", "IP", "HR/9", "xFIP", "Barrel%",
        "EV", "K/9", "BB/9", "Hard%", "ERA",
    ]]
