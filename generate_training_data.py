import os
import pandas as pd
import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"

ETA_MAG_LAT  = 25.0   # degrees geomagnetic latitude
ETA_ALT_KM   = 600.0  # km
# ─────────────────────────────────────────────────────────────────────────────


def datetime_to_jd(dt):
    year = dt.year; mon = dt.month; day = dt.day
    hr = dt.hour; minute = dt.minute
    sec = dt.second + dt.microsecond / 1e6
    if mon <= 2:
        year -= 1; mon += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25*(year+4716)) + int(30.6001*(mon+1)) + day + B - 1524.5
    fr = (hr + minute/60.0 + sec/3600.0) / 24.0
    return jd, fr


def eci_to_rsw(r_pred, v_pred, r_true):
    r_p = np.array(r_pred); v_p = np.array(v_pred); r_t = np.array(r_true)
    dr = r_t - r_p
    r_mag = np.linalg.norm(r_p)
    if r_mag == 0: return 0.0, 0.0, 0.0
    u_r = r_p / r_mag
    h = np.cross(r_p, v_p); h_mag = np.linalg.norm(h)
    if h_mag == 0: return 0.0, 0.0, 0.0
    u_w = h / h_mag
    u_s = np.cross(u_w, u_r)
    return float(np.dot(dr, u_r)), float(np.dot(dr, u_s)), float(np.dot(dr, u_w))


def teme_to_geodetic(r, jd, fr):
    t = (jd - 2451545.0 + fr) / 36525.0
    gmst = (280.46061837 + 360.98564736629*(jd - 2451545.0 + fr)
            + 0.000387933*t**2 - t**3/38710000.0)
    g = np.radians(gmst % 360.0)
    x = r[0]*np.cos(g) + r[1]*np.sin(g)
    y = -r[0]*np.sin(g) + r[1]*np.cos(g)
    z = r[2]
    a = 6378.137; f = 1/298.257223563; e2 = 2*f - f**2
    p = np.sqrt(x**2 + y**2)
    if p < 1e-6:
        return (90.0 if z > 0 else -90.0), 0.0, abs(z) - a*(1-f)
    lon = np.degrees(np.arctan2(y, x))
    lat_r = np.arctan2(z, p*(1-e2))
    for _ in range(5):
        sl = np.sin(lat_r); N = a/np.sqrt(1 - e2*sl**2)
        lat_r = np.arctan2(z + e2*N*sl, p)
    lat = np.degrees(lat_r)
    sl = np.sin(lat_r); N = a/np.sqrt(1 - e2*sl**2)
    alt = p/np.cos(lat_r) - N
    return lat, lon, alt


def get_geomagnetic_latitude(lat_deg, lon_deg):
    lat_p = np.radians(80.7); lon_p = np.radians(-72.7)
    lat_r = np.radians(lat_deg); lon_r = np.radians(lon_deg)
    s = np.sin(lat_r)*np.sin(lat_p) + np.cos(lat_r)*np.cos(lat_p)*np.cos(lon_r - lon_p)
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def get_local_time(utc_dt, lon_deg):
    h = utc_dt.hour + utc_dt.minute/60.0 + utc_dt.second/3600.0
    return (h + lon_deg/15.0) % 24.0


def magnetic_local_time(utc_dt, lon_deg):
    lt = get_local_time(utc_dt, lon_deg)
    return (lt + 12.0) % 24.0


def load_sw_daily(hist_path, dst_path):
    print("  Loading historical daily space weather...")
    hist = pd.read_csv(hist_path)
    date_col = hist.columns[0]
    hist[date_col] = pd.to_datetime(hist[date_col], errors='coerce')
    hist = hist.dropna(subset=[date_col]).set_index(date_col)
    hist.index = hist.index.normalize()

    kp_cols = [c for c in ['Kp1','Kp2','Kp3','Kp4','Kp5','Kp6','Kp7','Kp8']
               if c in hist.columns]
    if kp_cols:
        hist['Kp_index'] = hist[kp_cols].mean(axis=1)
    else:
        hist['Kp_index'] = 2.0

    if 'F10.7obs' in hist.columns:
        hist.rename(columns={'F10.7obs': 'F107'}, inplace=True)

    if 'F107' in hist.columns:
        hist['solar_cycle_phase'] = ((hist['F107'].clip(70, 230) - 70) / 160.0)
    else:
        hist['solar_cycle_phase'] = 0.5

    if os.path.exists(dst_path):
        print("  Loading Dst index...")
        dst = pd.read_csv(dst_path, parse_dates=['Datetime'])
        dst['date'] = dst['Datetime'].dt.normalize()
        dst_daily = dst.groupby('date')['Dst'].mean().rename('Dst')
        hist = hist.join(dst_daily, how='left')
        hist['Dst'] = hist['Dst'].fillna(0.0)
    else:
        hist['Dst'] = 0.0

    hist['Xray_short'] = 1e-7
    hist['Xray_long']  = 1e-7

    print(f"  SW daily range: {hist.index.min().date()} to {hist.index.max().date()}")
    return hist


def process_satellite(args):
    """
    Multiprocessing worker function.
    args is a tuple: (sat_df, sw_hist_dict)
    """
    sat_df, sw_hist_dict = args
    sat_df = sat_df.sort_values('Parsed_Epoch').reset_index(drop=True)
    results = []

    for i in range(len(sat_df)):
        row_A = sat_df.iloc[i]
        
        # Look ahead to find TLEs up to 30 days (720h) away
        for j in range(i + 1, min(i + 40, len(sat_df))):
            row_B = sat_df.iloc[j]

            t_A = row_A['Parsed_Epoch']
            t_B = row_B['Parsed_Epoch']
            dt_hours = (t_B - t_A).total_seconds() / 3600.0

            if dt_hours < 1.0:
                continue
            if dt_hours > 720.0:
                break  # TLEs are sorted, so any subsequent j will also be > 720h

            try:
                satrec_A = Satrec.twoline2rv(str(row_A['TLE_LINE1']), str(row_A['TLE_LINE2']))
                satrec_B = Satrec.twoline2rv(str(row_B['TLE_LINE1']), str(row_B['TLE_LINE2']))
            except Exception:
                continue

            # Validate orbital parameter ranges:
            # Eccentricity (ecco): [0.0, 1.0)
            # Inclination (inclo): [0.0, pi] in radians
            # BSTAR (bstar): [-10.0, 10.0]
            if not (0.0 <= satrec_A.ecco < 1.0 and 0.0 <= satrec_A.inclo <= np.pi and -10.0 <= satrec_A.bstar <= 10.0):
                continue
            if not (0.0 <= satrec_B.ecco < 1.0 and 0.0 <= satrec_B.inclo <= np.pi and -10.0 <= satrec_B.bstar <= 10.0):
                continue

            jd_B, fr_B = datetime_to_jd(t_B.to_pydatetime())

            e_true, r_true, v_true = satrec_B.sgp4(jd_B, fr_B)
            e_pred, r_pred, v_pred = satrec_A.sgp4(jd_B, fr_B)

            if e_true != 0 or e_pred != 0:
                continue

            err_radial, err_along, err_cross = eci_to_rsw(r_pred, v_pred, r_true)
            
            # Scale cleaning thresholds with propagation gap size (e.g. error grows up to 3000km at 30 days)
            limit_scale = 1.0 + (dt_hours / 24.0)
            if abs(err_along) > 1000.0 * limit_scale or abs(err_radial) > 300.0 * limit_scale or abs(err_cross) > 300.0 * limit_scale:
                continue

            lat, lon, alt = teme_to_geodetic(r_pred, jd_B, fr_B)

            # DATA CLEANING: Skip physically impossible altitudes (satellite reentered or coordinates bad)
            if alt < 100.0 or alt > 1000.0:
                continue

            magnetic_lat  = get_geomagnetic_latitude(lat, lon)
            local_time    = get_local_time(t_B.to_pydatetime(), lon)
            mag_lt        = magnetic_local_time(t_B.to_pydatetime(), lon)

            in_eta = abs(magnetic_lat) <= ETA_MAG_LAT and alt < ETA_ALT_KM
            eta_lat_intensity = max(0.0, 1.0 - abs(magnetic_lat) / ETA_MAG_LAT) if in_eta else 0.0
            eta_alt_intensity = max(0.0, 1.0 - (alt - 300.0) / (ETA_ALT_KM - 300.0)) if in_eta and alt > 300 else (1.0 if in_eta else 0.0)
            eta_intensity = (eta_lat_intensity + eta_alt_intensity) / 2.0

            date_key = t_B.normalize()
            if hasattr(date_key, 'tz_localize'):
                date_key_naive = date_key.tz_localize(None) if date_key.tzinfo is None else date_key.tz_convert(None)
            else:
                date_key_naive = date_key

            sw = sw_hist_dict.get(date_key_naive, None)
            if sw is None:
                continue  # Data Cleaning: Skip if no space weather match exists

            result = {
                'NORAD_CAT_ID':      row_A['NORAD_CAT_ID'],
                'dt_hours':          dt_hours,
                'BSTAR':             satrec_A.bstar,
                'INCLINATION':       satrec_A.inclo,
                'ECCENTRICITY':      satrec_A.ecco,
                'altitude':          alt,
                'latitude':          lat,
                'longitude':         lon,
                'magnetic_lat':      magnetic_lat,
                'local_time':        local_time,
                'magnetic_lt':       mag_lt,
                'in_eta':            int(in_eta),
                'eta_intensity':     eta_intensity,
                'err_radial':        err_radial,
                'err_along':         err_along,
                'err_cross':         err_cross,
                'Ap':                float(sw['Ap']),
                'F107':              float(sw['F107']),
                'Kp_index':          float(sw['Kp_index']),
                'Dst':               float(sw['Dst']),
                'solar_cycle_phase': float(sw['solar_cycle_phase']),
                'Xray_short':        float(sw['Xray_short']),
                'Xray_long':         float(sw['Xray_long']),
            }
            results.append(result)


    return results


def main():
    hist_path = os.path.join(BASE_DIR, 'compiled_historical_daily.csv')
    dst_path  = os.path.join(BASE_DIR, 'data', 'dst_index.csv')
    tle_path  = os.path.join(BASE_DIR, 'data', 'starlink_sample.csv')
    out_path  = os.path.join(BASE_DIR, 'ml_training_dataset.csv')

    if not os.path.exists(tle_path):
        print(f"TLE file not found: {tle_path}"); return
    if not os.path.exists(hist_path):
        print(f"Historical SW file not found: {hist_path}"); return

    # ── Load space weather ────────────────────────────────────────────────
    sw_hist = load_sw_daily(hist_path, dst_path)
    # Convert space weather df into dict for fast lookup inside workers
    sw_hist_dict = sw_hist.to_dict(orient='index')

    # ── Load TLEs ─────────────────────────────────────────────────────────
    print("Loading TLE data...")
    cols = ['NORAD_CAT_ID', 'EPOCH', 'TLE_LINE1', 'TLE_LINE2']
    tle_raw = pd.read_csv(tle_path, usecols=cols, on_bad_lines='skip')
    
    # 1. Check missing/unparseable epochs
    total_raw = len(tle_raw)
    tle_df = tle_raw.dropna(subset=['EPOCH', 'TLE_LINE1', 'TLE_LINE2'])
    dropped_missing = total_raw - len(tle_df)
    
    # 2. Convert timestamps to UTC
    tle_df['Parsed_Epoch'] = pd.to_datetime(tle_df['EPOCH'], utc=True, errors='coerce', format='mixed')
    tle_df = tle_df.dropna(subset=['Parsed_Epoch'])
    dropped_invalid_epoch = (total_raw - dropped_missing) - len(tle_df)
    
    # 3. DATA CLEANING: Remove TLEs with empty strings or bad lines
    tle_df = tle_df[tle_df['TLE_LINE1'].str.len() > 50]
    tle_df = tle_df[tle_df['TLE_LINE2'].str.len() > 50]
    
    # 4. Remove duplicate TLEs (same NORAD_CAT_ID and Epoch)
    before_dedup = len(tle_df)
    tle_df = tle_df.drop_duplicates(subset=['NORAD_CAT_ID', 'Parsed_Epoch'])
    dropped_duplicates = before_dedup - len(tle_df)

    print(f"Data Cleaning Summary:")
    print(f"  - Total raw records loaded: {total_raw:,}")
    print(f"  - Dropped due to missing data: {dropped_missing:,}")
    print(f"  - Dropped due to invalid epoch parsing: {dropped_invalid_epoch:,}")
    print(f"  - Dropped duplicate TLE entries: {dropped_duplicates:,}")
    print(f"  - Final clean records to propagate: {len(tle_df):,} across {tle_df['NORAD_CAT_ID'].nunique():,} satellites.")

    # ── Process satellites in Parallel ────────────────────────────────────
    print("\nCalculating SGP4 residuals + ETA features in parallel...")
    grouped = [group for _, group in tle_df.groupby('NORAD_CAT_ID')]
    worker_args = [(group, sw_hist_dict) for group in grouped]

    num_cores = max(1, cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for parallel processing...")

    all_results = []
    with Pool(num_cores) as pool:
        results = pool.map(process_satellite, worker_args)
        for res in results:
            all_results.extend(res)

    if len(all_results) == 0:
        print("ERROR: No training samples generated. Check date overlap.")
        return

    final_df = pd.DataFrame(all_results)
    
    # DATA CLEANING: Strict outlier filtering (remove extreme error values)
    # Filter 3-sigma residuals to prevent learning bad radar tracking jumps
    for col in ['err_radial', 'err_along', 'err_cross']:
        mu, sig = final_df[col].mean(), final_df[col].std()
        final_df = final_df[(final_df[col] >= mu - 3*sig) & (final_df[col] <= mu + 3*sig)]

    final_df = final_df.dropna()

    n_total = len(final_df)
    n_eta   = final_df['in_eta'].sum()
    print(f"\nGenerated {n_total:,} cleaned training samples ({n_eta:,} ETA = "
          f"{100*n_eta/n_total:.1f}%)")
    
    final_df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")


if __name__ == "__main__":
    main()
