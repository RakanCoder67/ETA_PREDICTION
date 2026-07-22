import os
import threading
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(BASE_DIR, "data", "starlink_sample.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Starlink ETA Orbit Prediction", version="1.0.0")

# Serve model images as /images/<filename>
app.mount("/images", StaticFiles(directory=MODELS_DIR), name="images")

# Path to the HTML file
HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")

# ---------------------------------------------------------------------------
# Load ML corrector once at startup (expensive — do it once)
# ---------------------------------------------------------------------------
corrector = None
sample_df = None
available_norads = []
live_starlink_df = None          # Full live constellation from CelesTrak
live_starlink_updated_at = None  # Last fetch timestamp

try:
    from starlink_eta_corrector import StarlinkETACorrector, datetime_to_jd, teme_to_geodetic
    print("Loading ML corrector…")
    corrector = StarlinkETACorrector()
    if corrector.models is None:
        print("WARNING: ML models not found — predictions will be SGP4 only.")
except Exception as ex:
    print(f"WARNING: Could not load StarlinkETACorrector: {ex}")

try:
    if os.path.exists(SAMPLE_CSV):
        print("Loading satellite sample database…")
        sample_df = pd.read_csv(SAMPLE_CSV)
        sample_df["EPOCH"] = pd.to_datetime(sample_df["EPOCH"], utc=True)
        available_norads = sorted(sample_df["NORAD_CAT_ID"].unique().tolist())
        print(f"Loaded {len(available_norads)} satellites.")
    else:
        print(f"WARNING: {SAMPLE_CSV} not found.")
except Exception as ex:
    print(f"WARNING: Could not load sample CSV: {ex}")

# ---------------------------------------------------------------------------
# Live Starlink TLE fetcher (CelesTrak — no auth required)
# ---------------------------------------------------------------------------
CELESTRAK_URL = "https://celestrak.org/SOCRATES/query.php?GROUP=starlink&FORMAT=tle"
LIVE_TLE_CACHE = os.path.join(BASE_DIR, "data", "starlink_live_tle.txt")


def _parse_epoch_from_tle1(line1: str) -> datetime:
    """Parse the epoch from TLE line 1 into a UTC datetime."""
    try:
        epoch_str = line1[18:32].strip()
        year2 = int(epoch_str[:2])
        year = 2000 + year2 if year2 < 57 else 1900 + year2
        day_of_year = float(epoch_str[2:])
        epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
        return epoch
    except Exception:
        return datetime.now(timezone.utc)


def _parse_norad_from_tle1(line1: str) -> int:
    try:
        return int(line1[2:7].strip())
    except Exception:
        return 0


def fetch_live_starlink_tles() -> pd.DataFrame | None:
    """Download current Starlink TLEs from CelesTrak and return a DataFrame."""
    global live_starlink_df, live_starlink_updated_at
    try:
        print("Fetching live Starlink TLEs from CelesTrak…")
        resp = requests.get(CELESTRAK_URL, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()
        # Cache locally for offline fallback
        os.makedirs(os.path.dirname(LIVE_TLE_CACHE), exist_ok=True)
        with open(LIVE_TLE_CACHE, "w") as f:
            f.write(text)
    except Exception as ex:
        print(f"WARNING: CelesTrak fetch failed ({ex}). Trying local cache…")
        if os.path.exists(LIVE_TLE_CACHE):
            with open(LIVE_TLE_CACHE) as f:
                text = f.read().strip()
        else:
            print("No local cache available. Live ETA scan will use sample DB.")
            return None

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rows = []
    for i in range(0, len(lines) - 2, 3):
        name = lines[i]
        l1   = lines[i + 1]
        l2   = lines[i + 2]
        if not (l1.startswith('1') and l2.startswith('2')):
            continue
        norad  = _parse_norad_from_tle1(l1)
        epoch  = _parse_epoch_from_tle1(l1)
        rows.append({
            "NORAD_CAT_ID": norad,
            "OBJECT_NAME":  name,
            "TLE_LINE1":    l1,
            "TLE_LINE2":    l2,
            "EPOCH":        epoch,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)
    live_starlink_df = df
    live_starlink_updated_at = datetime.now(timezone.utc)
    print(f"Live TLE fetch complete: {len(df):,} Starlink satellites loaded.")
    return df


def _daily_tle_refresh():
    """Background thread: refresh TLEs every 24 hours."""
    while True:
        time.sleep(86400)   # 24 hours
        try:
            fetch_live_starlink_tles()
        except Exception as ex:
            print(f"Daily TLE refresh failed: {ex}")


# Initial fetch at startup
fetch_live_starlink_tles()

# Start daily background refresh thread
_refresh_thread = threading.Thread(target=_daily_tle_refresh, daemon=True)
_refresh_thread.start()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    norad_id: int
    custom_hours: float | None = None
    live_mode: bool = False

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    sw = {}
    if corrector is not None and corrector.sw_df is not None:
        try:
            latest = corrector.sw_df.iloc[-1]
            sw = {
                "Kp": round(float(latest.get("Kp_index", 2.0)), 2),
                "F107": round(float(latest.get("F107", 150.0)), 1),
                "Ap": int(latest.get("Ap", 7)),
            }
        except Exception:
            sw = {"Kp": 2.0, "F107": 150.0, "Ap": 7}

    return {
        "status": "ok",
        "models_loaded": corrector is not None and corrector.models is not None,
        "satellites_in_db": len(available_norads),
        "space_weather": sw,
        "sample_norads": available_norads[:10],
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if sample_df is None or len(sample_df) == 0:
        raise HTTPException(status_code=503, detail="Satellite database not loaded.")

    sat_df = sample_df[sample_df["NORAD_CAT_ID"] == req.norad_id].sort_values("EPOCH")
    if len(sat_df) == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"NORAD ID {req.norad_id} not found in database. "
                f"Try one of: {available_norads[:5]}"
            ),
        )

    if corrector is None:
        raise HTTPException(status_code=503, detail="ML corrector not initialised.")

    from sgp4.api import Satrec

    latest = sat_df.iloc[-1]
    sat_name = str(latest["OBJECT_NAME"])
    tle_epoch = latest["EPOCH"]
    tle_epoch_str = tle_epoch.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        tle1 = latest["TLE_LINE1"]
        tle2 = latest["TLE_LINE2"]
        # Zero BSTAR for long-horizon propagation (avoids decay blow-up)
        tle1_zeroed = tle1[:53] + " 00000-0" + tle1[61:]
        satrec = Satrec.twoline2rv(tle1_zeroed, tle2)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"TLE parse error: {ex}")

    # Current space weather snapshot
    now_utc = datetime.now(timezone.utc)
    sw_snap = corrector.get_space_weather_at(now_utc)
    space_weather = {
        "Kp": float(sw_snap.get("Kp_index", 2.0)),
        "F107": float(sw_snap.get("F107", 150.0)),
        "Ap": int(sw_snap.get("Ap", 7)),
        "Dst": float(sw_snap.get("Dst", 0.0)),
    }

    # Check ETA status at epoch (or NOW in live mode)
    try:
        ref_time = now_utc if req.live_mode else tle_epoch.to_pydatetime()
        jd0, fr0 = datetime_to_jd(ref_time)
        _, r0, v0 = satrec.sgp4(jd0, fr0)
        lat0, lon0, alt0 = teme_to_geodetic(np.array(r0), jd0, fr0)
        from starlink_eta_corrector import get_geomagnetic_latitude
        mag_lat0 = get_geomagnetic_latitude(lat0, lon0)
        in_eta_now = bool(corrector.is_in_eta_region(mag_lat0, alt0))
    except Exception:
        in_eta_now = False

    # Build trajectory: key horizons
    if req.live_mode:
        base_horizons = [0.0, 24.0, 48.0, 72.0, 168.0, 360.0, 720.0]
    else:
        base_horizons = [24.0, 48.0, 72.0, 168.0, 360.0, 720.0]

    if req.custom_hours is not None and req.custom_hours > 0:
        base_horizons.append(float(req.custom_hours))
    
    # Sort unique horizons
    HORIZONS_H = sorted(list(set(base_horizons)))
    trajectory = []
    custom_point = None

    for h in HORIZONS_H:
        if req.live_mode:
            t_target = now_utc + timedelta(hours=h)
        else:
            t_target = tle_epoch.to_pydatetime() + timedelta(hours=h)

        jd, fr = datetime_to_jd(t_target)

        # SGP4 baseline
        err, r_sgp4, v_sgp4 = satrec.sgp4(jd, fr)
        if err != 0:
            continue

        r_sgp4 = np.array(r_sgp4)

        # ML corrected (always propagate relative to standard start_epoch so BSTAR/time drift is model-consistent)
        try:
            r_ml, _, info = corrector.propagate_and_correct(satrec, tle_epoch.to_pydatetime(), t_target)
        except Exception:
            r_ml = r_sgp4
            info = {"in_eta": False, "lat": 0.0, "lon": 0.0, "alt": 0.0, "mag_lat": 0.0}

        lat = info.get("lat", 0.0)
        lon = info.get("lon", 0.0)
        alt = info.get("alt", 0.0)
        mag_lat = info.get("mag_lat", 0.0)
        in_eta = bool(info.get("in_eta", False))
        correction_km = float(np.linalg.norm(r_ml - r_sgp4))

        is_custom = bool(req.custom_hours is not None and abs(float(h) - float(req.custom_hours)) < 1e-4)

        pt_data = {
            "offset_hours": round(float(h), 2) if not float(h).is_integer() else int(h),
            "epoch_utc": t_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sgp4": [round(float(r_sgp4[0]), 3), round(float(r_sgp4[1]), 3), round(float(r_sgp4[2]), 3)],
            "ml_corrected": [round(float(r_ml[0]), 3), round(float(r_ml[1]), 3), round(float(r_ml[2]), 3)],
            "lat": round(float(lat), 3),
            "lon": round(float(lon), 3),
            "alt": round(float(alt), 2),
            "mag_lat": round(float(mag_lat), 3),
            "in_eta": bool(in_eta),
            "correction_km": round(float(correction_km), 4),
            "is_custom": is_custom,
        }

        trajectory.append(pt_data)
        if is_custom:
            custom_point = pt_data

    return JSONResponse(content={
        "satellite_name": sat_name,
        "norad_id": int(req.norad_id),
        "tle_epoch": tle_epoch_str,
        "space_weather": space_weather,
        "in_eta_now": bool(in_eta_now),
        "custom_point": custom_point,
        "live_mode": bool(req.live_mode),
        "trajectory": trajectory,
    })


@app.get("/eta_now")
async def eta_now():
    """Return all Starlink satellites currently inside the ETA region."""
    if corrector is None:
        raise HTTPException(status_code=503, detail="ML corrector not ready.")

    # Use full live constellation; fall back to sample DB if fetch hasn't run yet
    scan_df = live_starlink_df if live_starlink_df is not None else sample_df
    if scan_df is None:
        raise HTTPException(status_code=503, detail="No satellite data available.")

    from sgp4.api import Satrec

    now_utc = datetime.now(timezone.utc)
    jd_now, fr_now = datetime_to_jd(now_utc)

    results = []
    # One row per satellite (latest TLE)
    latest_tles = scan_df.sort_values("EPOCH").groupby("NORAD_CAT_ID").last().reset_index()

    for _, row in latest_tles.iterrows():
        try:
            satrec = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            e, r, v = satrec.sgp4(jd_now, fr_now)
            if e != 0:
                continue
            r = np.array(r)
            lat, lon, alt = teme_to_geodetic(r, jd_now, fr_now)
            from starlink_eta_corrector import get_geomagnetic_latitude
            mag_lat = get_geomagnetic_latitude(lat, lon)
            if not corrector.is_in_eta_region(mag_lat, alt):
                continue

            # ML-corrected position
            try:
                r_ml, _, info = corrector.propagate_and_correct(
                    satrec, row["EPOCH"].to_pydatetime(), now_utc
                )
                correction_km = float(np.linalg.norm(np.array(r_ml) - r))
            except Exception:
                correction_km = 0.0

            results.append({
                "norad_id":       int(row["NORAD_CAT_ID"]),
                "name":           str(row["OBJECT_NAME"]),
                "lat":            round(float(lat), 2),
                "lon":            round(float(lon), 2),
                "alt":            round(float(alt), 1),
                "mag_lat":        round(float(mag_lat), 2),
                "correction_km":  round(correction_km, 4),
                "tle_epoch":      row["EPOCH"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        except Exception:
            continue

    results.sort(key=lambda x: abs(x["mag_lat"]))
    return JSONResponse(content={
        "checked_at":   now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_in_eta": len(results),
        "total_scanned": len(latest_tles),
        "tle_source":   "CelesTrak (live)" if live_starlink_df is not None else "Sample DB (fallback)",
        "satellites":   results,
    })

