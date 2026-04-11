# ⚾ MLB Home Run Predictor

A **Random Forest** machine learning system that predicts which MLB hitters are
most likely to hit a home run on any given day. Results are published as a
static website on **GitHub Pages**, updated daily via **GitHub Actions**.

---

## How it works

| Layer | Detail |
|---|---|
| **Batter features** | HR%, ISO, BB%, K%, barrel%, exit velocity, hard-hit%, launch angle, pull%, HR/FB, xSLG |
| **Pitcher features** | HR/9, xFIP, barrel% allowed, exit velo allowed, K/9, BB/9 |
| **Bullpen feature** | Team HR/9 (covers late-game relief exposure) |
| **Park feature** | HR park factor (1.27 at Coors → 0.87 at loanDepot) + elevation |
| **Weather features** | Temperature, wind factor (blowing out/in toward CF), humidity |
| **ML model** | `RandomForestClassifier` (400 trees) trained on 2 full MLB seasons of Statcast game-level data |
| **Data source** | `pybaseball` → FanGraphs + Baseball Savant (Statcast) |
| **Lineups / starters** | MLB Stats API via `statsapi` |
| **Weather** | [Open-Meteo](https://open-meteo.com) – free, no API key needed |

---

## Quick start (local)

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/mlb-hr-predictor.git
cd mlb-hr-predictor
pip install -r requirements.txt

# 2. First run: train model + generate today's predictions
python main.py

# 3. On subsequent days: just predict (skips re-training)
python main.py --predict

# 4. Force re-train with fresh data
python main.py --train
```

The script creates a `cache/` directory to store downloaded data and the
trained model.  Large Statcast parquet files (1-2 GB per season) are cached
locally but excluded from git.

---

## GitHub Pages setup

1. **Fork / clone** this repo.
2. Enable **GitHub Pages** in *Settings → Pages → Source: `docs/` folder, `main` branch*.
3. The included **GitHub Actions workflow** (`.github/workflows/daily_predict.yml`)
   runs every day at **11:00 AM ET** and pushes an updated `docs/index.html`.
4. Your predictions page will be live at:
   `https://YOUR_USERNAME.github.io/mlb-hr-predictor/`

### First-run note
On the first GitHub Actions run, the model must be trained from scratch
(~2 full seasons of Statcast data).  This can take **20-40 minutes** the
first time; subsequent daily runs are much faster (incremental data updates
only).  The Actions cache keeps the downloaded Statcast files between runs.

---

## File structure

```
mlb-hr-predictor/
├── main.py                      ← Orchestrator (run this)
├── config.py                    ← Global settings
├── requirements.txt
├── .gitignore
├── src/
│   ├── ballparks.py             ← 30 MLB parks: coords, HR factors, CF bearing
│   ├── data_fetcher.py          ← pybaseball wrapper with smart caching
│   ├── features.py              ← Feature engineering
│   ├── model.py                 ← RandomForest train / save / load
│   ├── predictor.py             ← Today's games + lineup handling
│   ├── site_generator.py        ← HTML generation
│   └── weather.py               ← Open-Meteo weather API
├── cache/                       ← Auto-created; large files gitignored
├── docs/
│   └── index.html               ← Generated site (committed to git)
└── .github/
    └── workflows/
        └── daily_predict.yml    ← Scheduled Actions workflow
```

---

## Adjusting the model

Edit `config.py`:

| Setting | Default | Effect |
|---|---|---|
| `TRAINING_YEARS` | last 2 seasons | More years → more training data |
| `RF_PARAMS["n_estimators"]` | 400 | More trees → slower but more accurate |
| `RF_PARAMS["max_depth"]` | 12 | Deeper → may overfit |
| `MIN_PA_CURRENT_SEASON` | 30 | Filter out players with too few at-bats |
| `TOP_N` | 30 | How many predictions to display |

---

## Disclaimer

For entertainment purposes only.  This tool does not constitute gambling
advice.  Baseball is famously unpredictable.
