"""
src/weather.py
Fetches current / forecast weather for a park location using Open-Meteo.
No API key required.
"""
import requests
from datetime import datetime, date
import math

from config import OPEN_METEO_URL


def _mph(ms: float) -> float:
    """Metres-per-second → miles-per-hour"""
    return round(ms * 2.23694, 1)


def get_weather(lat: float, lon: float, game_date: str = None) -> dict:
    """
    Return weather dict for a given lat/lon.

    Parameters
    ----------
    lat, lon    : park GPS coordinates
    game_date   : 'YYYY-MM-DD'; defaults to today

    Returns
    -------
    dict with keys:
        temp_f          – temperature in °F
        wind_speed_mph  – sustained wind speed in mph
        wind_dir_deg    – wind direction in degrees (meteorological: FROM which direction)
        humidity_pct    – relative humidity %
        precip_mm       – precipitation probability (%)
        condition       – short text description
        is_dome_weather – False (dome flag is handled in ballparks.py)
    """
    if game_date is None:
        game_date = date.today().isoformat()

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "hourly":           "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,precipitation_probability,weathercode",
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "timezone":         "auto",
        "start_date":       game_date,
        "end_date":         game_date,
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [weather] API error: {exc} – using neutral defaults")
        return _neutral_weather()

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    # Pick the afternoon hour closest to 1 PM local (index 13)
    idx = 13 if len(times) > 13 else len(times) // 2

    def safe(key, default=0.0):
        vals = hourly.get(key, [])
        return vals[idx] if idx < len(vals) else default

    temp_f        = safe("temperature_2m",            72.0)
    humidity      = safe("relativehumidity_2m",        50.0)
    wind_speed    = safe("windspeed_10m",               0.0)
    wind_dir      = safe("winddirection_10m",           0.0)
    precip_prob   = safe("precipitation_probability",   0.0)
    weather_code  = int(safe("weathercode",             0))

    return {
        "temp_f":         round(temp_f, 1),
        "wind_speed_mph": round(wind_speed, 1),
        "wind_dir_deg":   round(wind_dir, 1),
        "humidity_pct":   round(humidity, 1),
        "precip_pct":     round(precip_prob, 1),
        "condition":      _wmo_description(weather_code),
    }


def _neutral_weather() -> dict:
    return {
        "temp_f":         72.0,
        "wind_speed_mph": 5.0,
        "wind_dir_deg":   0.0,
        "humidity_pct":   50.0,
        "precip_pct":     0.0,
        "condition":      "Unknown",
    }


# WMO Weather Code → short description
_WMO = {
    0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow",
    80: "Showers", 81: "Heavy Showers", 82: "Violent Showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ Hail", 99: "Severe Thunderstorm",
}

def _wmo_description(code: int) -> str:
    return _WMO.get(code, f"Code {code}")
