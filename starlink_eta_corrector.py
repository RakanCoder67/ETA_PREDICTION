import os
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS84

# Coordinate conversion utilities
def datetime_to_jd(dt):
    time_tuple = dt.timetuple()
    year = time_tuple.tm_year
    mon = time_tuple.tm_mon
    day = time_tuple.tm_mday
    hr = time_tuple.tm_hour
    minute = time_tuple.tm_min
    sec = time_tuple.tm_sec + dt.microsecond / 1e6
    if mon <= 2:
        year -= 1
        mon += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (mon + 1)) + day + B - 1524.5
    fr = (hr + minute / 60.0 + sec / 3600.0) / 24.0
    return jd, fr

def teme_to_geodetic(r, jd, fr):
    t = (jd - 2451545.0 + fr) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0 + fr) + 0.000387933 * (t**2) - (t**3) / 38710000.0
    gmst_rad = np.radians(gmst % 360.0)
    
    x_ecef = r[0] * np.cos(gmst_rad) + r[1] * np.sin(gmst_rad)
    y_ecef = -r[0] * np.sin(gmst_rad) + r[1] * np.cos(gmst_rad)
    z_ecef = r[2]
    
    a = 6378.137 # km
    f = 1.0 / 298.257223563
    e2 = 2 * f - f**2
    
    p = np.sqrt(x_ecef**2 + y_ecef**2)
    if p < 1e-6:
        lon = 0.0
        lat = 90.0 if z_ecef > 0 else -90.0
        alt = np.abs(z_ecef) - a * (1 - f)
        return lat, lon, alt
        
    lon_rad = np.arctan2(y_ecef, x_ecef)
    lon = np.degrees(lon_rad)
    
    lat_rad = np.arctan2(z_ecef, p * (1 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat_rad)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        lat_rad = np.arctan2(z_ecef + e2 * N * sin_lat, p)
        
    lat = np.degrees(lat_rad)
    sin_lat = np.sin(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    alt = p / np.cos(lat_rad) - N
    
    return lat, lon, alt

def get_geomagnetic_latitude(lat_deg, lon_deg):
    lat_p = np.radians(80.7)
    lon_p = np.radians(-72.7)
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    sin_lat_m = np.sin(lat_r) * np.sin(lat_p) + np.cos(lat_r) * np.cos(lat_p) * np.cos(lon_r - lon_p)
    sin_lat_m = np.clip(sin_lat_m, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_lat_m))

def get_local_time(utc_datetime, lon_deg):
    utc_hour = utc_datetime.hour + utc_datetime.minute / 60.0 + utc_datetime.second / 3600.0 + utc_datetime.microsecond / 3.6e9
    local_hour = (utc_hour + lon_deg / 15.0) % 24.0
    return local_hour

def get_rsw_basis(r_pred, v_pred):
    r_p = np.array(r_pred)
    v_p = np.array(v_pred)
    
    r_mag = np.linalg.norm(r_p)
    if r_mag == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    u_r = r_p / r_mag
    
    h = np.cross(r_p, v_p)
    h_mag = np.linalg.norm(h)
    if h_mag == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    u_w = h / h_mag
    
    u_s = np.cross(u_w, u_r)
    return u_r, u_s, u_w

class StarlinkETACorrector:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_path = os.path.join(self.base_dir, 'models', 'sgp4_correction_models.pkl')
        self.features_path = os.path.join(self.base_dir, 'models', 'model_features.pkl')
        self.sw_path = os.path.join(self.base_dir, 'compiled_recent_1m.csv')
        self.hist_path = os.path.join(self.base_dir, 'compiled_historical_daily.csv')
        self.dst_path = os.path.join(self.base_dir, 'data', 'dst_index.csv')
        
        self.models = None
        self.features = None
        self.sw_df = None
        self.sw_hist = None
        
        self.load_resources()
        
    def load_resources(self):
        if os.path.exists(self.models_path) and os.path.exists(self.features_path):
            print("Loading machine learning models...")
            self.models = joblib.load(self.models_path)
            self.features = joblib.load(self.features_path)
        else:
            print("Warning: ML models not found! Run train_ml_model.py first to enable corrections.")
            
        if os.path.exists(self.sw_path):
            print("Loading compiled space weather dataset...")
            self.sw_df = pd.read_csv(self.sw_path)
            self.sw_df['Datetime'] = pd.to_datetime(self.sw_df['Datetime'], utc=True)
            self.sw_df.set_index('Datetime', inplace=True)
        else:
            print("Warning: Space weather data not found! Run fetch_live_data.py first.")
            
        if os.path.exists(self.hist_path):
            print("Loading historical daily space weather for TLE overlap fallback...")
            hist = pd.read_csv(self.hist_path)
            date_col = hist.columns[0]
            hist[date_col] = pd.to_datetime(hist[date_col], errors='coerce')
            hist = hist.dropna(subset=[date_col]).set_index(date_col)
            hist.index = hist.index.normalize()
            kp_cols = [c for c in ['Kp1','Kp2','Kp3','Kp4','Kp5','Kp6','Kp7','Kp8'] if c in hist.columns]
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
            if os.path.exists(self.dst_path):
                dst = pd.read_csv(self.dst_path, parse_dates=['Datetime'])
                dst['date'] = dst['Datetime'].dt.normalize()
                dst_daily = dst.groupby('date')['Dst'].mean().rename('Dst')
                hist = hist.join(dst_daily, how='left')
                hist['Dst'] = hist['Dst'].fillna(0.0)
            else:
                hist['Dst'] = 0.0
            hist['Xray_short'] = 1e-7
            hist['Xray_long']  = 1e-7
            self.sw_hist = hist

    def get_space_weather_at(self, target_time):
        target_dt = pd.to_datetime(target_time, utc=True)
        date_key = target_dt.normalize()
        if hasattr(date_key, 'tz_localize'):
            date_key_naive = date_key.tz_localize(None) if date_key.tzinfo is None else date_key.tz_convert(None)
        else:
            date_key_naive = date_key

        # First try historical daily fallback if it is a past date
        if self.sw_hist is not None and date_key_naive in self.sw_hist.index:
            row = self.sw_hist.loc[date_key_naive]
            return {
                'Ap': float(row.get('Ap', 7.0)),
                'F107': float(row.get('F107', 150.0)),
                'Kp_index': float(row.get('Kp_index', 2.0)),
                'Dst': float(row.get('Dst', 0.0)),
                'solar_cycle_phase': float(row.get('solar_cycle_phase', 0.5)),
                'Xray_short': 1e-7,
                'Xray_long': 1e-7
            }
            
        # Try minute resolution if within recent window
        if self.sw_df is not None:
            t_min = target_dt.floor('min')
            if t_min in self.sw_df.index:
                row = self.sw_df.loc[t_min]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return {
                    'Ap': float(row.get('Ap', 7.0)),
                    'F107': float(row.get('F107', 150.0)),
                    'Kp_index': float(row.get('Kp_index', 2.0)),
                    'Dst': float(row.get('Dst', 0.0)),
                    'solar_cycle_phase': float(row.get('solar_cycle_phase', 0.5) if 'solar_cycle_phase' in row else 0.5),
                    'Xray_short': float(row.get('Xray_flux_short', 1e-7)),
                    'Xray_long': float(row.get('Xray_flux_long', 1e-7))
                }
            
        return {'Ap': 7.0, 'F107': 150.0, 'Kp_index': 2.0, 'Dst': 0.0, 'solar_cycle_phase': 0.5, 'Xray_short': 1e-7, 'Xray_long': 1e-7}
            
    def is_in_eta_region(self, mag_lat, alt):
        return abs(mag_lat) <= 25.0 and alt < 600.0

    def propagate_and_correct(self, satrec, start_epoch, target_time):
        jd, fr = datetime_to_jd(target_time)
        e, r_pred, v_pred = satrec.sgp4(jd, fr)
        
        if e != 0:
            raise RuntimeError(f"SGP4 propagation failed with error code {e}")
            
        r_pred = np.array(r_pred)
        v_pred = np.array(v_pred)
        
        lat, lon, alt = teme_to_geodetic(r_pred, jd, fr)
        mag_lat = get_geomagnetic_latitude(lat, lon)
        local_time = get_local_time(target_time, lon)
        
        # Calculate MLT and ETA intensity features
        # MLT approximation:
        mag_lt = (local_time + 12.0) % 24.0
        
        dt_hours = (target_time - start_epoch).total_seconds() / 3600.0
        in_eta = self.is_in_eta_region(mag_lat, alt)
        
        eta_lat_intensity = max(0.0, 1.0 - abs(mag_lat) / 25.0) if in_eta else 0.0
        eta_alt_intensity = max(0.0, 1.0 - (alt - 300.0) / 300.0) if in_eta and alt > 300 else (1.0 if in_eta else 0.0)
        eta_intensity = (eta_lat_intensity + eta_alt_intensity) / 2.0
        
        if self.models is None:
            return r_pred, v_pred, {
                'lat': lat, 'lon': lon, 'alt': alt, 'mag_lat': mag_lat, 
                'local_time': local_time, 'in_eta': in_eta, 
                'corrected': False, 'msg': 'ML Models not loaded'
            }
            
        sw = self.get_space_weather_at(target_time)
        
        feat_dict = {
            'dt_hours': dt_hours,
            'BSTAR': satrec.bstar,
            'INCLINATION': satrec.inclo,
            'ECCENTRICITY': satrec.ecco,
            'altitude': alt,
            'latitude': lat,
            'longitude': lon,
            'magnetic_lat': mag_lat,
            'local_time': local_time,
            'magnetic_lt': mag_lt,
            'eta_intensity': eta_intensity,
            'in_eta': int(in_eta),
            'Ap': sw['Ap'],
            'F107': sw['F107'],
            'Kp_index': sw['Kp_index'],
            'Dst': sw['Dst'],
            'solar_cycle_phase': sw['solar_cycle_phase'],
            'Xray_short': sw['Xray_short'],
            'Xray_long': sw['Xray_long']
        }
        
        feat_df = pd.DataFrame([feat_dict])[self.features]
        
        # Physical time-scaling factor to ensure smooth, monotonic error growth starting at 0 at dt=0
        if dt_hours <= 0:
            time_scale = 0.0
        elif dt_hours <= 24.0:
            time_scale = (dt_hours / 24.0) ** 1.15
        else:
            time_scale = 1.0 + 0.08 * np.log1p(dt_hours - 24.0)

        # Raw tree predictions
        raw_radial = float(self.models['err_radial'].predict(feat_df)[0])
        raw_along = float(self.models['err_along'].predict(feat_df)[0])
        raw_cross = float(self.models['err_cross'].predict(feat_df)[0])

        # Apply time scale
        dr_radial = raw_radial * time_scale
        dr_along = raw_along * time_scale
        dr_cross = raw_cross * time_scale

        # Physical upper bound cap on magnitude based on time horizon to avoid unphysical outliers
        max_allowed_km = max(0.02, 0.025 * dt_hours) if dt_hours <= 24.0 else (0.6 + 0.015 * dt_hours)
        total_mag = float(np.sqrt(dr_radial**2 + dr_along**2 + dr_cross**2))
        if total_mag > max_allowed_km and total_mag > 0:
            scale = max_allowed_km / total_mag
            dr_radial *= scale
            dr_along *= scale
            dr_cross *= scale
        
        u_r, u_s, u_w = get_rsw_basis(r_pred, v_pred)
        r_corrected = r_pred + dr_radial * u_r + dr_along * u_s + dr_cross * u_w
        
        info = {
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'mag_lat': mag_lat,
            'local_time': local_time,
            'in_eta': in_eta,
            'corrected': True,
            'predicted_res': {
                'radial': dr_radial,
                'along': dr_along,
                'cross': dr_cross
            },
            'space_weather': sw
        }
        
        return r_corrected, v_pred, info

def test_on_satellites(norad_ids=None, max_samples=50):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tle_path = os.path.join(base_dir, 'data', 'starlink_history.csv')
    
    if not os.path.exists(tle_path):
        print(f"Error: {tle_path} not found.")
        return
        
    print("Loading TLE database...")
    tle_df = pd.read_csv(tle_path)
    
    corrector = StarlinkETACorrector()
    if corrector.models is None:
        print("Models not available. Please run train_ml_model.py first.")
        return
        
    # Group by satellite
    grouped = tle_df.groupby('NORAD_CAT_ID')
    
    total_sgp4_err_along = []
    total_hybrid_err_along = []
    
    total_sgp4_err_total = []
    total_hybrid_err_total = []
    
    eta_sgp4_err_along = []
    eta_hybrid_err_along = []
    
    samples_count = 0
    
    print("\nEvaluating correction model on actual consecutive TLE data...")
    print(f"{'NORAD ID':<9} | {'dt (hrs)':<8} | {'In ETA?':<7} | {'SGP4 Err (km)':<13} | {'Hybrid Err (km)':<15} | {'Along-track Imp.':<16}")
    print("-" * 85)
    
    for norad_id, sat_df in grouped:
        if norad_ids and norad_id not in norad_ids:
            continue
            
        sat_df = sat_df.sort_values('EPOCH')
        if len(sat_df) < 2:
            continue
            
        for i in range(len(sat_df) - 1):
            row_A = sat_df.iloc[i]
            row_B = sat_df.iloc[i+1]
            
            t_A = pd.to_datetime(row_A['EPOCH'], utc=True)
            t_B = pd.to_datetime(row_B['EPOCH'], utc=True)
            
            dt_hours = (t_B - t_A).total_seconds() / 3600.0
            
            # Use same window as training
            if dt_hours > 168.0 or dt_hours < 1.0:
                continue
                
            try:
                satrec_A = Satrec.twoline2rv(row_A['TLE_LINE1'], row_A['TLE_LINE2'])
                satrec_B = Satrec.twoline2rv(row_B['TLE_LINE1'], row_B['TLE_LINE2'])
            except:
                continue
                
            jd_B, fr_B = datetime_to_jd(t_B)
            e_true, r_true, v_true = satrec_B.sgp4(jd_B, fr_B)
            if e_true != 0:
                continue
                
            # Propagate and Correct
            try:
                r_corrected, v_pred, info = corrector.propagate_and_correct(satrec_A, t_A, t_B)
            except Exception as e:
                continue
                
            # SGP4 baseline error
            e_pred, r_pred, v_pred = satrec_A.sgp4(jd_B, fr_B)
            
            r_pred = np.array(r_pred)
            r_true = np.array(r_true)
            
            # Calculate RSW basis relative to predicted state
            u_r, u_s, u_w = get_rsw_basis(r_pred, v_pred)
            
            # ECI positional error vectors
            dr_sgp4 = r_true - r_pred
            dr_hybrid = r_true - r_corrected
            
            # Along-track error (projection on u_s)
            along_err_sgp4 = abs(np.dot(dr_sgp4, u_s))
            along_err_hybrid = abs(np.dot(dr_hybrid, u_s))
            
            # Total positional 3D error magnitudes
            sgp4_mag = np.linalg.norm(dr_sgp4)
            hybrid_mag = np.linalg.norm(dr_hybrid)
            
            total_sgp4_err_along.append(along_err_sgp4)
            total_hybrid_err_along.append(along_err_hybrid)
            
            total_sgp4_err_total.append(sgp4_mag)
            total_hybrid_err_total.append(hybrid_mag)
            
            is_eta = info['in_eta']
            if is_eta:
                eta_sgp4_err_along.append(along_err_sgp4)
                eta_hybrid_err_along.append(along_err_hybrid)
                
            samples_count += 1
            
            # Print sample statistics
            if samples_count <= max_samples:
                eta_str = "YES" if is_eta else "NO"
                imp_pct = (along_err_sgp4 - along_err_hybrid) / along_err_sgp4 * 100 if along_err_sgp4 > 0 else 0
                print(f"{norad_id:<9} | {dt_hours:<8.1f} | {eta_str:<7} | {sgp4_mag:<13.3f} | {hybrid_mag:<15.3f} | {imp_pct:>+.1f}%")
                
            if samples_count >= max_samples * 2:
                # Limit evaluation to keep output concise
                break
                
    # Print overall performance summary
    if len(total_sgp4_err_along) > 0:
        mean_sgp4_along = np.mean(total_sgp4_err_along)
        mean_hybrid_along = np.mean(total_hybrid_err_along)
        mean_sgp4_total = np.mean(total_sgp4_err_total)
        mean_hybrid_total = np.mean(total_hybrid_err_total)
        
        along_imp = (mean_sgp4_along - mean_hybrid_along) / mean_sgp4_along * 100
        total_imp = (mean_sgp4_total - mean_hybrid_total) / mean_sgp4_total * 100
        
        print("\n" + "="*50)
        print("OVERALL PERFORMANCE COMPARISON")
        print("="*50)
        print(f"Total Samples Evaluated: {len(total_sgp4_err_along)}")
        print(f"Mean Along-Track Error (SGP4):   {mean_sgp4_along:.3f} km")
        print(f"Mean Along-Track Error (Hybrid): {mean_hybrid_along:.3f} km")
        print(f"Along-Track Error Reduction:     {along_imp:.1f}%")
        print(f"Mean 3D Position Error (SGP4):   {mean_sgp4_total:.3f} km")
        print(f"Mean 3D Position Error (Hybrid): {mean_hybrid_total:.3f} km")
        print(f"Total 3D Error Reduction:        {total_imp:.1f}%")
        
        if len(eta_sgp4_err_along) > 0:
            mean_sgp4_eta = np.mean(eta_sgp4_err_along)
            mean_hybrid_eta = np.mean(eta_hybrid_err_along)
            eta_imp = (mean_sgp4_eta - mean_hybrid_eta) / mean_sgp4_eta * 100
            print(f"\n--- Equatorial Thermosphere Anomaly (ETA) Transits ---")
            print(f"ETA Samples Evaluated:           {len(eta_sgp4_err_along)}")
            print(f"Mean Along-Track Error (SGP4):   {mean_sgp4_eta:.3f} km")
            print(f"Mean Along-Track Error (Hybrid): {mean_hybrid_eta:.3f} km")
            print(f"ETA Along-Track Error Reduction: {eta_imp:.1f}%")
        print("="*50)
        
def main():
    parser = argparse.ArgumentParser(description="Starlink Orbit Prediction & Correction System (ETA-Optimized)")
    parser.add_argument('--norad', type=int, help='NORAD catalog ID of a Starlink satellite to simulate.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the ML corrector model against all compiled consecutive TLE pairs.')
    
    args = parser.parse_args()
    
    if args.evaluate:
        test_on_satellites()
    elif args.norad:
        corrector = StarlinkETACorrector()
        
        # Load satellite TLE
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tle_path = os.path.join(base_dir, 'data', 'starlink_history.csv')
        if not os.path.exists(tle_path):
            print("Error: TLE history not found. Run fetch_live_data.py first.")
            return
            
        df = pd.read_csv(tle_path)
        sat_df = df[df['NORAD_CAT_ID'] == args.norad].sort_values('EPOCH')
        
        if len(sat_df) < 1:
            print(f"Error: Satellite NORAD ID {args.norad} not found in database.")
            return
            
        row = sat_df.iloc[-1]
        print(f"\nSimulating propagation of {row['OBJECT_NAME']} (NORAD {args.norad})")
        print(f"Epoch of TLE elements: {row['EPOCH']}")
        
        satrec = Satrec.twoline2rv(row['TLE_LINE1'], row['TLE_LINE2'])
        start_epoch = pd.to_datetime(row['EPOCH'], utc=True)
        
        # Propagate for 24 hours at 2-hour intervals
        print(f"\nPropagating forward 24 hours from epoch...")
        print(f"{'Time (+hrs)':<11} | {'Geodetic Lat/Lon':<20} | {'Mag Lat':<8} | {'Alt (km)':<8} | {'In ETA?':<7} | {'Predicted Along Error':<21}")
        print("-" * 90)
        
        for hrs in range(0, 26, 2):
            target_time = start_epoch + timedelta(hours=hrs)
            r_corr, v_pred, info = corrector.propagate_and_correct(satrec, start_epoch, target_time)
            
            lat_lon_str = f"{info['lat']:.1f}° / {info['lon']:.1f}°"
            eta_str = "YES" if info['in_eta'] else "NO"
            along_pred = info['predicted_res']['along'] if info['corrected'] else 0.0
            
            print(f"{hrs:<11.1f} | {lat_lon_str:<20} | {info['mag_lat']:<8.1f} | {info['alt']:<8.1f} | {eta_str:<7} | {along_pred:>+18.4f} km")
            
    else:
        # Default run: run evaluation on a few sample satellites
        test_on_satellites(max_samples=20)

if __name__ == "__main__":
    main()
