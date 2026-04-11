"""
src/ballparks.py
All 30 MLB ballparks with:
  - GPS coordinates (for weather API)
  - HR park factor (>1.0 = HR-friendly; league-average ≈ 1.0)
  - CF bearing from home plate (degrees, 0° = North) for wind calculations
  - Elevation (feet)
"""

# ── Master ballpark table ─────────────────────────────────────────────────────
# Keys are the home-team abbreviation used by statsapi / pybaseball
PARKS = {
    # ── AL East ──────────────────────────────────────────────────────────────
    "BAL": {
        "name": "Oriole Park at Camden Yards",
        "city": "Baltimore, MD",
        "lat": 39.2838, "lon": -76.6218,
        "elevation_ft": 20,
        "hr_factor": 1.03,
        "cf_bearing": 35,          # NNE
    },
    "BOS": {
        "name": "Fenway Park",
        "city": "Boston, MA",
        "lat": 42.3467, "lon": -71.0972,
        "elevation_ft": 20,
        "hr_factor": 0.92,         # Small park but quirky walls
        "cf_bearing": 50,
    },
    "NYY": {
        "name": "Yankee Stadium",
        "city": "Bronx, NY",
        "lat": 40.8296, "lon": -73.9262,
        "elevation_ft": 55,
        "hr_factor": 1.10,
        "cf_bearing": 55,
    },
    "TB":  {
        "name": "Tropicana Field",
        "city": "St. Petersburg, FL",
        "lat": 27.7683, "lon": -82.6534,
        "elevation_ft": 44,
        "hr_factor": 0.95,         # Dome – no wind factor
        "cf_bearing": 0,
        "dome": True,
    },
    "TOR": {
        "name": "Rogers Centre",
        "city": "Toronto, ON",
        "lat": 43.6414, "lon": -79.3894,
        "elevation_ft": 250,
        "hr_factor": 1.08,
        "cf_bearing": 10,
        "dome": True,
    },
    # ── AL Central ───────────────────────────────────────────────────────────
    "CWS": {
        "name": "Guaranteed Rate Field",
        "city": "Chicago, IL",
        "lat": 41.8300, "lon": -87.6339,
        "elevation_ft": 595,
        "hr_factor": 1.07,
        "cf_bearing": 25,
    },
    "CLE": {
        "name": "Progressive Field",
        "city": "Cleveland, OH",
        "lat": 41.4962, "lon": -81.6852,
        "elevation_ft": 650,
        "hr_factor": 0.99,
        "cf_bearing": 20,
    },
    "DET": {
        "name": "Comerica Park",
        "city": "Detroit, MI",
        "lat": 42.3390, "lon": -83.0485,
        "elevation_ft": 600,
        "hr_factor": 0.89,         # Very HR-unfriendly
        "cf_bearing": 30,
    },
    "KC":  {
        "name": "Kauffman Stadium",
        "city": "Kansas City, MO",
        "lat": 39.0517, "lon": -94.4803,
        "elevation_ft": 1014,
        "hr_factor": 0.96,
        "cf_bearing": 15,
    },
    "MIN": {
        "name": "Target Field",
        "city": "Minneapolis, MN",
        "lat": 44.9817, "lon": -93.2783,
        "elevation_ft": 830,
        "hr_factor": 1.00,
        "cf_bearing": 330,         # NNW
    },
    # ── AL West ──────────────────────────────────────────────────────────────
    "HOU": {
        "name": "Minute Maid Park",
        "city": "Houston, TX",
        "lat": 29.7573, "lon": -95.3555,
        "elevation_ft": 43,
        "hr_factor": 1.02,
        "cf_bearing": 10,
        "dome": True,              # Retractable
    },
    "LAA": {
        "name": "Angel Stadium",
        "city": "Anaheim, CA",
        "lat": 33.8003, "lon": -117.8827,
        "elevation_ft": 160,
        "hr_factor": 0.96,
        "cf_bearing": 45,
    },
    "OAK": {
        "name": "Oakland Coliseum",
        "city": "Oakland, CA",
        "lat": 37.7516, "lon": -122.2005,
        "elevation_ft": 23,
        "hr_factor": 0.88,         # Very HR-unfriendly
        "cf_bearing": 330,
    },
    "SEA": {
        "name": "T-Mobile Park",
        "city": "Seattle, WA",
        "lat": 47.5914, "lon": -122.3325,
        "elevation_ft": 43,
        "hr_factor": 0.93,
        "cf_bearing": 15,
        "dome": True,              # Retractable
    },
    "TEX": {
        "name": "Globe Life Field",
        "city": "Arlington, TX",
        "lat": 32.7473, "lon": -97.0822,
        "elevation_ft": 551,
        "hr_factor": 1.05,
        "cf_bearing": 10,
        "dome": True,              # Retractable
    },
    # ── NL East ──────────────────────────────────────────────────────────────
    "ATL": {
        "name": "Truist Park",
        "city": "Cumberland, GA",
        "lat": 33.8908, "lon": -84.4677,
        "elevation_ft": 1050,
        "hr_factor": 1.08,
        "cf_bearing": 25,
    },
    "MIA": {
        "name": "loanDepot park",
        "city": "Miami, FL",
        "lat": 25.7781, "lon": -80.2197,
        "elevation_ft": 10,
        "hr_factor": 0.87,         # Very HR-unfriendly
        "cf_bearing": 350,
        "dome": True,              # Retractable
    },
    "NYM": {
        "name": "Citi Field",
        "city": "Flushing, NY",
        "lat": 40.7571, "lon": -73.8458,
        "elevation_ft": 20,
        "hr_factor": 0.93,
        "cf_bearing": 40,
    },
    "PHI": {
        "name": "Citizens Bank Park",
        "city": "Philadelphia, PA",
        "lat": 39.9061, "lon": -75.1665,
        "elevation_ft": 20,
        "hr_factor": 1.10,         # Very HR-friendly
        "cf_bearing": 15,
    },
    "WSH": {
        "name": "Nationals Park",
        "city": "Washington, DC",
        "lat": 38.8730, "lon": -77.0074,
        "elevation_ft": 20,
        "hr_factor": 1.04,
        "cf_bearing": 10,
    },
    # ── NL Central ───────────────────────────────────────────────────────────
    "CHC": {
        "name": "Wrigley Field",
        "city": "Chicago, IL",
        "lat": 41.9484, "lon": -87.6553,
        "elevation_ft": 595,
        "hr_factor": 1.02,         # Variable; wind-dependent
        "cf_bearing": 70,          # ENE
    },
    "CIN": {
        "name": "Great American Ball Park",
        "city": "Cincinnati, OH",
        "lat": 39.0979, "lon": -84.5082,
        "elevation_ft": 483,
        "hr_factor": 1.16,         # Most HR-friendly in NL
        "cf_bearing": 20,
    },
    "MIL": {
        "name": "American Family Field",
        "city": "Milwaukee, WI",
        "lat": 43.0280, "lon": -87.9712,
        "elevation_ft": 635,
        "hr_factor": 1.03,
        "cf_bearing": 15,
        "dome": True,              # Retractable
    },
    "PIT": {
        "name": "PNC Park",
        "city": "Pittsburgh, PA",
        "lat": 40.4469, "lon": -80.0057,
        "elevation_ft": 730,
        "hr_factor": 0.94,
        "cf_bearing": 340,
    },
    "STL": {
        "name": "Busch Stadium",
        "city": "St. Louis, MO",
        "lat": 38.6226, "lon": -90.1928,
        "elevation_ft": 465,
        "hr_factor": 0.95,
        "cf_bearing": 20,
    },
    # ── NL West ──────────────────────────────────────────────────────────────
    "ARI": {
        "name": "Chase Field",
        "city": "Phoenix, AZ",
        "lat": 33.4453, "lon": -112.0667,
        "elevation_ft": 1090,
        "hr_factor": 1.05,
        "cf_bearing": 350,
        "dome": True,              # Retractable
    },
    "COL": {
        "name": "Coors Field",
        "city": "Denver, CO",
        "lat": 39.7559, "lon": -104.9942,
        "elevation_ft": 5200,      # Mile High!
        "hr_factor": 1.27,         # Most HR-friendly in MLB
        "cf_bearing": 20,
    },
    "LAD": {
        "name": "Dodger Stadium",
        "city": "Los Angeles, CA",
        "lat": 34.0739, "lon": -118.2400,
        "elevation_ft": 512,
        "hr_factor": 0.96,
        "cf_bearing": 310,
    },
    "SD":  {
        "name": "Petco Park",
        "city": "San Diego, CA",
        "lat": 32.7076, "lon": -117.1570,
        "elevation_ft": 20,
        "hr_factor": 0.88,         # Very HR-unfriendly
        "cf_bearing": 330,
    },
    "SF":  {
        "name": "Oracle Park",
        "city": "San Francisco, CA",
        "lat": 37.7786, "lon": -122.3893,
        "elevation_ft": 10,
        "hr_factor": 0.88,
        "cf_bearing": 0,           # Wind often blows IN
    },
}

# Aliases used by statsapi that differ from standard abbreviations
TEAM_ALIASES = {
    "ANA": "LAA",
    "KCA": "KC",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBA": "TB",
    "WSN": "WSH",
    "WAS": "WSH",
    "CHW": "CWS",
}


def get_park(team_abbrev: str) -> dict:
    """Return park info dict for a team abbreviation (with alias resolution)."""
    abbrev = TEAM_ALIASES.get(team_abbrev.upper(), team_abbrev.upper())
    park = PARKS.get(abbrev)
    if park is None:
        # Fallback: league-average neutral park
        park = {
            "name": "Unknown Park",
            "city": "Unknown",
            "lat": 39.5, "lon": -98.3,
            "elevation_ft": 600,
            "hr_factor": 1.0,
            "cf_bearing": 0,
        }
    park["team_abbrev"] = abbrev
    return park


def is_dome(team_abbrev: str) -> bool:
    park = get_park(team_abbrev)
    return park.get("dome", False)


def wind_factor(wind_speed_mph: float, wind_dir_deg: float, cf_bearing: float) -> float:
    """
    Returns a wind factor in [-1, 1]:
      +1 → wind perfectly blowing OUT to CF (HR-helpful)
      -1 → wind perfectly blowing IN from CF (HR-hurting)
       0 → crosswind or no wind
    """
    import math
    # Angle between wind direction and CF direction
    diff = (wind_dir_deg - cf_bearing + 360) % 360
    # diff=0 → wind FROM north while CF is north → wind blowing OUT (tailwind)
    # diff=180 → wind blowing IN from CF (headwind)
    # Use cosine: cos(0°)=1 (tailwind), cos(180°)=-1 (headwind)
    cosine = math.cos(math.radians(diff))
    # Scale by wind speed (cap at 25 mph for normalization)
    speed_factor = min(wind_speed_mph / 25.0, 1.0)
    return round(cosine * speed_factor, 3)
