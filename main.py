#!/usr/bin/env python3
"""
main.py – MLB Home Run Predictor
=================================
Usage:
  python main.py               # train + predict (default)
  python main.py --train       # (re)train the model only
  python main.py --predict     # predict today using saved model
  python main.py --train --predict  # same as default
"""

import argparse
import os
import sys
from datetime import datetime
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="MLB HR Predictor – Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train",   action="store_true", help="Train / retrain the model")
    parser.add_argument("--predict", action="store_true", help="Generate today's predictions")
    args = parser.parse_args()

    # Default: do both
    if not args.train and not args.predict:
        args.train   = True
        args.predict = True

    # ── Ensure output directories exist ──────────────────────────────────────
    from config import CACHE_DIR, DOCS_DIR, ALL_YEARS
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Lazy imports (gives cleaner error messages if packages missing)
    try:
        from src.data_fetcher  import DataFetcher
        from src.model         import HRModel
        from src.predictor     import Predictor
        import src.site_generator as site_gen
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

    fetcher = DataFetcher()
    model   = HRModel()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 – DATA + TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    if args.train:
        _banner("DATA COLLECTION")

        print("\n▸ Statcast (pitch-level, cached by season)…")
        statcast_data = fetcher.get_statcast_all_years()

        print("\n▸ Batting stats…")
        batting_data = {}
        for yr in ALL_YEARS:
            df = fetcher.get_batting_stats(yr)
            if df.empty:
                print(f"  [fallback] FanGraphs blocked – deriving batting stats {yr} from Statcast…")
                df = fetcher.derive_batting_stats(statcast_data.get(yr, pd.DataFrame()))
            batting_data[yr] = df

        print("\n▸ Pitching stats…")
        pitching_data = {}
        for yr in ALL_YEARS:
            df = fetcher.get_pitching_stats(yr)
            if df.empty:
                print(f"  [fallback] FanGraphs blocked – deriving pitching stats {yr} from Statcast…")
                df = fetcher.derive_pitching_stats(statcast_data.get(yr, pd.DataFrame()))
            pitching_data[yr] = df

        _banner("FEATURE ENGINEERING")
        from src.features import FeatureBuilder
        builder = FeatureBuilder()
        X, y, feat_names = builder.build_training_features(
            statcast_data, batting_data, pitching_data
        )

        _banner("MODEL TRAINING")
        model.train(X, y, feat_names)
        model.save()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 – PREDICT TODAY
    # ══════════════════════════════════════════════════════════════════════════
    if args.predict:
        _banner("TODAY'S PREDICTIONS")

        if not args.train:          # Load previously saved model
            model.load()

        predictor   = Predictor(fetcher)
        predictions = predictor.predict_today(model)

        _banner("GENERATING WEBSITE")
        site_gen.generate(predictions, model_info=model.get_info())

        if predictions:
            print(f"\n{'─'*55}")
            print(f"  {'RANK':<5} {'PLAYER':<25} {'PROB':>6}")
            print(f"{'─'*55}")
            for rank, p in enumerate(predictions[:10], 1):
                print(f"  #{rank:<4} {p['player_name']:<25} {p['hr_pct_display']:>6}")
            print(f"{'─'*55}")
            print(f"  Full rankings → docs/index.html")
        else:
            print("  No predictions generated (no games today or data unavailable).")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
