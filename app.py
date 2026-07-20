import os
import numpy as np
import pandas as pd
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
# Request / Response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    norad_id: int

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

    # Check ETA status at epoch
    try:
        jd0, fr0 = datetime_to_jd(tle_epoch.to_pydatetime())
        _, r0, v0 = satrec.sgp4(jd0, fr0)
        lat0, lon0, alt0 = teme_to_geodetic(np.array(r0), jd0, fr0)
        from starlink_eta_corrector import get_geomagnetic_latitude
        mag_lat0 = get_geomagnetic_latitude(lat0, lon0)
        in_eta_now = bool(corrector.is_in_eta_region(mag_lat0, alt0))
    except Exception:
        in_eta_now = False

    # Build trajectory: key horizons
    HORIZONS_H = [24, 48, 72, 168, 360, 720]  # hours
    trajectory = []

    for h in HORIZONS_H:
        t_target = tle_epoch.to_pydatetime() + timedelta(hours=h)
        jd, fr = datetime_to_jd(t_target)

        # SGP4 baseline
        err, r_sgp4, v_sgp4 = satrec.sgp4(jd, fr)
        if err != 0:
            continue

        r_sgp4 = np.array(r_sgp4)

        # ML corrected
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

        trajectory.append({
            "offset_hours": int(h),
            "epoch_utc": t_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sgp4": [round(float(r_sgp4[0]), 3), round(float(r_sgp4[1]), 3), round(float(r_sgp4[2]), 3)],
            "ml_corrected": [round(float(r_ml[0]), 3), round(float(r_ml[1]), 3), round(float(r_ml[2]), 3)],
            "lat": round(float(lat), 3),
            "lon": round(float(lon), 3),
            "alt": round(float(alt), 2),
            "mag_lat": round(float(mag_lat), 3),
            "in_eta": bool(in_eta),
            "correction_km": round(float(correction_km), 4),
        })

    return JSONResponse(content={
        "satellite_name": sat_name,
        "norad_id": int(req.norad_id),
        "tle_epoch": tle_epoch_str,
        "space_weather": space_weather,
        "in_eta_now": bool(in_eta_now),
        "trajectory": trajectory,
    })