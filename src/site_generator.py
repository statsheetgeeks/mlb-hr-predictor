"""
src/site_generator.py
Generates docs/index.html – the static GitHub Pages site.
"""

import os
import json
from datetime import datetime

from config import DOCS_DIR, TOP_N


def _wind_arrow(deg: float) -> str:
    """Return a Unicode arrow roughly pointing the wind direction."""
    arrows = ["↑","↗","→","↘","↓","↙","←","↖"]
    idx = int((deg + 22.5) / 45) % 8
    return arrows[idx]


def _bar(prob: float) -> str:
    """SVG probability bar (0-1 range)."""
    pct   = round(prob * 100, 1)
    color = (
        "#22c55e" if prob >= 0.15 else
        "#eab308" if prob >= 0.10 else
        "#f97316" if prob >= 0.07 else
        "#94a3b8"
    )
    return (
        f'<div class="bar-wrap">'
        f'  <div class="bar-fill" style="width:{min(pct*4,100):.1f}%;background:{color};"></div>'
        f'  <span class="bar-label">{pct}%</span>'
        f'</div>'
    )


def _wind_badge(wf: float) -> str:
    if abs(wf) < 0.1:
        label, cls = "→ Cross", "badge-neutral"
    elif wf > 0:
        label, cls = f"↑ Out ({wf:+.2f})", "badge-good"
    else:
        label, cls = f"↓ In ({wf:+.2f})", "badge-bad"
    return f'<span class="badge {cls}">{label}</span>'


def _park_badge(factor: float) -> str:
    if factor >= 1.10:
        label, cls = f"★ {factor:.2f}", "badge-good"
    elif factor <= 0.92:
        label, cls = f"▼ {factor:.2f}", "badge-bad"
    else:
        label, cls = f"● {factor:.2f}", "badge-neutral"
    return f'<span class="badge {cls}">{label}</span>'


def generate(predictions: list[dict], model_info: dict):
    """Write docs/index.html."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    now  = datetime.now()
    date_str = now.strftime("%A, %B %-d, %Y")
    time_str = now.strftime("%-I:%M %p CT")

    # ── Model sidebar info ───────────────────────────────────────────────────
    cv_auc  = model_info.get("cv_roc_auc_mean", 0)
    trained = model_info.get("trained_at", "")[:10]
    n_samp  = f"{model_info.get('n_samples', 0):,}"
    hr_rate = f"{model_info.get('hr_rate', 0)*100:.2f}%"
    top_feats = model_info.get("feature_importances", {})
    top5 = list(top_feats.items())[:5]

    feat_rows = "\n".join(
        f'<tr><td>{nm.replace("_"," ").title()}</td>'
        f'<td><div class="mini-bar" style="width:{v*600:.0f}px"></div>{v:.3f}</td></tr>'
        for nm, v in top5
    )

    # ── Prediction table rows ────────────────────────────────────────────────
    if predictions:
        table_rows = ""
        for rank, p in enumerate(predictions, 1):
            dome_note = " 🏟" if p.get("is_dome") else ""
            wind_info = (
                f'{_wind_badge(p["wind_factor"])}'
                if not p.get("is_dome") else
                '<span class="badge badge-neutral">Dome</span>'
            )
            table_rows += f"""
            <tr>
              <td class="rank">#{rank}</td>
              <td class="player-cell">
                <span class="player-name">{p['player_name']}</span>
                <span class="team-tag">{p['team_abbr']}</span>
              </td>
              <td>{p['opponent']}</td>
              <td class="pitcher-cell">{p['opp_starter']}</td>
              <td class="venue-cell">{p['venue']}{dome_note}</td>
              <td>{_park_badge(p['park_hr_factor'])}</td>
              <td class="weather-cell">
                {p['weather_temp']}°F<br>
                {p['weather_cond']}<br>
                {wind_info}
              </td>
              <td class="prob-cell">{_bar(p['hr_probability'])}</td>
            </tr>"""

        content = f"""
        <div class="table-wrap">
          <table id="predictions">
            <thead>
              <tr>
                <th>#</th>
                <th>Player</th>
                <th>Opp</th>
                <th>Starter</th>
                <th>Venue</th>
                <th>Park Factor</th>
                <th>Weather</th>
                <th>HR Probability</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>"""
    else:
        content = """
        <div class="no-games">
          <span class="no-games-icon">⚾</span>
          <h2>No games today</h2>
          <p>Check back on a game day!</p>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚾ MLB HR Predictor – {date_str}</title>
  <style>
    /* ── Reset & base ────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
    }}
    a {{ color: #60a5fa; text-decoration: none; }}

    /* ── Header ─────────────────────────────────────────────────── */
    header {{
      background: linear-gradient(135deg, #1e3a5f 0%, #0f2a45 100%);
      padding: 2rem 2rem 1.5rem;
      border-bottom: 2px solid #1e40af;
    }}
    .header-inner {{
      max-width: 1300px;
      margin: 0 auto;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    header h1 {{
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #f0f9ff;
    }}
    header h1 span {{ color: #38bdf8; }}
    .header-meta {{
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.6;
    }}
    .header-meta strong {{ color: #cbd5e1; }}

    /* ── Layout ──────────────────────────────────────────────────── */
    .layout {{
      max-width: 1300px;
      margin: 2rem auto;
      padding: 0 1.5rem 4rem;
      display: grid;
      grid-template-columns: 1fr 260px;
      gap: 2rem;
      align-items: start;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ order: -1; }}
    }}

    /* ── Sidebar ─────────────────────────────────────────────────── */
    .sidebar {{ display: flex; flex-direction: column; gap: 1.5rem; }}
    .card {{
      background: #1e2433;
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 1.25rem;
    }}
    .card h3 {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #60a5fa;
      margin-bottom: 1rem;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }}
    .stat-item {{ text-align: center; }}
    .stat-val {{
      font-size: 1.4rem;
      font-weight: 800;
      color: #38bdf8;
      display: block;
    }}
    .stat-lbl {{
      font-size: 0.7rem;
      color: #64748b;
      text-transform: uppercase;
    }}

    /* Feature importance mini-bars */
    .mini-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
    .mini-table td {{ padding: 3px 4px; color: #94a3b8; }}
    .mini-table td:first-child {{ white-space: nowrap; max-width: 130px; overflow: hidden; text-overflow: ellipsis; }}
    .mini-bar {{
      display: inline-block;
      height: 6px;
      background: #3b82f6;
      border-radius: 3px;
      vertical-align: middle;
      margin-right: 4px;
    }}

    /* Legend */
    .legend {{ display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.8rem; }}
    .leg-row {{ display: flex; align-items: center; gap: 0.5rem; }}
    .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

    /* ── Main table ──────────────────────────────────────────────── */
    .section-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: #f1f5f9;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .section-title .count-badge {{
      background: #1e40af;
      color: #93c5fd;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 20px;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    thead tr {{ background: #1a2535; }}
    thead th {{
      padding: 10px 12px;
      text-align: left;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #64748b;
      border-bottom: 2px solid #2d3748;
      white-space: nowrap;
    }}
    tbody tr {{
      border-bottom: 1px solid #1e2433;
      transition: background 0.15s;
    }}
    tbody tr:hover {{ background: #1e2d45; }}
    td {{
      padding: 10px 12px;
      vertical-align: middle;
      color: #cbd5e1;
    }}
    td.rank {{
      font-weight: 700;
      color: #475569;
      font-size: 0.8rem;
      width: 36px;
    }}
    /* Top 3 gold/silver/bronze */
    tbody tr:nth-child(1) td.rank {{ color: #fbbf24; }}
    tbody tr:nth-child(2) td.rank {{ color: #94a3b8; }}
    tbody tr:nth-child(3) td.rank {{ color: #c2855a; }}

    .player-cell {{ white-space: nowrap; }}
    .player-name {{ font-weight: 700; color: #f1f5f9; margin-right: 6px; }}
    .team-tag {{
      background: #1e3a5f;
      color: #60a5fa;
      font-size: 0.65rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .pitcher-cell {{ color: #94a3b8; max-width: 140px; }}
    .venue-cell {{ color: #94a3b8; font-size: 0.8rem; max-width: 140px; }}
    .weather-cell {{ font-size: 0.8rem; line-height: 1.6; }}
    .prob-cell {{ min-width: 150px; }}

    /* ── Probability bar ─────────────────────────────────────────── */
    .bar-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .bar-fill {{
      height: 8px;
      border-radius: 4px;
      min-width: 2px;
      transition: width 0.3s;
    }}
    .bar-label {{
      font-weight: 700;
      font-size: 0.85rem;
      color: #e2e8f0;
      white-space: nowrap;
    }}

    /* ── Badges ──────────────────────────────────────────────────── */
    .badge {{
      display: inline-block;
      font-size: 0.68rem;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 20px;
      white-space: nowrap;
    }}
    .badge-good    {{ background: #14532d; color: #4ade80; }}
    .badge-bad     {{ background: #450a0a; color: #f87171; }}
    .badge-neutral {{ background: #1e293b; color: #94a3b8; }}

    /* ── No games ────────────────────────────────────────────────── */
    .no-games {{
      text-align: center;
      padding: 5rem 2rem;
      color: #475569;
    }}
    .no-games-icon {{ font-size: 4rem; display: block; margin-bottom: 1rem; }}
    .no-games h2 {{ font-size: 1.5rem; color: #64748b; }}

    /* ── Footer ──────────────────────────────────────────────────── */
    footer {{
      text-align: center;
      padding: 2rem;
      font-size: 0.75rem;
      color: #334155;
      border-top: 1px solid #1e2433;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>⚾ MLB <span>HR Predictor</span></h1>
        <div class="header-meta">
          <strong>{date_str}</strong> &nbsp;·&nbsp; Updated {time_str}<br>
          Random Forest model · Statcast + FanGraphs · Open-Meteo weather
        </div>
      </div>
    </div>
  </header>

  <div class="layout">
    <!-- ── Main content ── -->
    <main>
      <div class="section-title">
        Today's Top Home Run Candidates
        <span class="count-badge">{len(predictions)} players</span>
      </div>
      {content}
    </main>

    <!-- ── Sidebar ── -->
    <aside class="sidebar">

      <div class="card">
        <h3>Model Performance</h3>
        <div class="stat-grid">
          <div class="stat-item">
            <span class="stat-val">{cv_auc:.3f}</span>
            <span class="stat-lbl">CV ROC-AUC</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">{n_samp}</span>
            <span class="stat-lbl">Training rows</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">{hr_rate}</span>
            <span class="stat-lbl">HR rate</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">{trained}</span>
            <span class="stat-lbl">Trained</span>
          </div>
        </div>
      </div>

      <div class="card">
        <h3>Top Features</h3>
        <table class="mini-table">
          {feat_rows}
        </table>
      </div>

      <div class="card">
        <h3>Probability Legend</h3>
        <div class="legend">
          <div class="leg-row"><div class="leg-dot" style="background:#22c55e"></div>≥ 15% – Elite target</div>
          <div class="leg-row"><div class="leg-dot" style="background:#eab308"></div>10–15% – Strong candidate</div>
          <div class="leg-row"><div class="leg-dot" style="background:#f97316"></div>7–10% – Worth watching</div>
          <div class="leg-row"><div class="leg-dot" style="background:#94a3b8"></div>&lt; 7% – Long shot</div>
        </div>
      </div>

      <div class="card">
        <h3>Data Sources</h3>
        <div style="font-size:0.78rem;color:#64748b;line-height:1.8;">
          🗃 Statcast (pybaseball)<br>
          📊 FanGraphs batting/pitching<br>
          🌤 Open-Meteo weather API<br>
          ⚾ MLB Stats API (lineups)<br>
          🏟 30-park HR park factors
        </div>
      </div>

    </aside>
  </div>

  <footer>
    For entertainment purposes only &nbsp;·&nbsp;
    Data via <a href="https://github.com/jldbc/pybaseball">pybaseball</a>,
    <a href="https://open-meteo.com">Open-Meteo</a>,
    <a href="https://statsapi.mlb.com">MLB Stats API</a> &nbsp;·&nbsp;
    Generated {now.strftime("%Y-%m-%d %H:%M")} UTC
  </footer>
</body>
</html>"""

    out_path = os.path.join(DOCS_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Site written → {out_path}")
