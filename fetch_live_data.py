import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

def download_file(url, save_path):
    print(f"Downloading {url}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # Save raw content
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def compile_live_space_weather():
    base_dir = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
    
    # 1. Download live JSON files from NOAA SWPC
    kp_url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    f107_url = "https://services.swpc.noaa.gov/products/10cm-flux-30-day.json"
    
    kp_path = os.path.join(base_dir, 'data', 'live_kp_1m.json')
    xray_path = os.path.join(base_dir, 'data', 'live_xray_7d.json')
    f107_path = os.path.join(base_dir, 'data', 'live_f107_30d.json')
    
    download_file(kp_url, kp_path)
    download_file(xray_url, xray_path)
    download_file(f107_url, f107_path)
    
    # 2. Parse and Compile space weather data
    print("Compiling live space weather data...")
    
    # Load Kp Index (1-minute tags)
    if os.path.exists(kp_path):
        try:
            with open(kp_path, 'r') as f:
                kp_data = json.load(f)
            df_kp = pd.DataFrame(kp_data)
            df_kp['Datetime'] = pd.to_datetime(df_kp['time_tag'], utc=True)
            df_kp.set_index('Datetime', inplace=True)
            df_kp = df_kp[['kp_index', 'estimated_kp']]
        except Exception as e:
            print(f"Error parsing Kp index: {e}")
            df_kp = pd.DataFrame()
    else:
        df_kp = pd.DataFrame()
        
    # Load GOES X-ray Flux
    if os.path.exists(xray_path):
        try:
            with open(xray_path, 'r') as f:
                xray_data = json.load(f)
            df_xray = pd.DataFrame(xray_data)
            df_xray['Datetime'] = pd.to_datetime(df_xray['time_tag'], utc=True)
            # Pivot by energy bands
            df_xray_pivot = df_xray.pivot_table(index='Datetime', columns='energy', values='flux', aggfunc='mean')
            df_xray_pivot.rename(columns={'0.05-0.4nm': 'Xray_flux_short', '0.1-0.8nm': 'Xray_flux_long'}, inplace=True)
        except Exception as e:
            print(f"Error parsing X-ray flux: {e}")
            df_xray_pivot = pd.DataFrame()
    else:
        df_xray_pivot = pd.DataFrame()
        
    # Load F10.7 Flux (Daily values in the 30-day product)
    # The format can be an array of objects: {"time_tag": "YYYY-MM-DD HH:MM:SS", "flux": F10.7}
    # Or a header-first array of arrays if it hasn't migrated yet. Let's make it robust!
    f107_values = {}
    if os.path.exists(f107_path):
        try:
            with open(f107_path, 'r') as f:
                f107_data = json.load(f)
            
            # Check if it is list of objects or list of lists
            if isinstance(f107_data, list) and len(f107_data) > 0:
                if isinstance(f107_data[0], dict):
                    # List of objects format
                    for item in f107_data:
                        t_str = item.get('time_tag')
                        val = item.get('flux')
                        if t_str and val is not None:
                            dt = pd.to_datetime(t_str, utc=True)
                            f107_values[dt] = float(val)
                elif isinstance(f107_data[0], list):
                    # Old list of lists format: index 0 is headers, subsequent are data
                    headers = f107_data[0]
                    t_idx = headers.index('time_tag') if 'time_tag' in headers else 0
                    f_idx = headers.index('flux') if 'flux' in headers else 1
                    for item in f107_data[1:]:
                        if len(item) > max(t_idx, f_idx):
                            dt = pd.to_datetime(item[t_idx], utc=True)
                            f107_values[dt] = float(item[f_idx])
        except Exception as e:
            print(f"Error parsing F10.7 data: {e}")
            
    df_f107 = pd.DataFrame(list(f107_values.items()), columns=['Datetime', 'F107']).set_index('Datetime')
    
    # 3. Join everything into a 1-minute grid
    if not df_kp.empty and not df_xray_pivot.empty:
        start_t = min(df_kp.index.min(), df_xray_pivot.index.min())
        end_t = max(df_kp.index.max(), df_xray_pivot.index.max())
    elif not df_kp.empty:
        start_t = df_kp.index.min()
        end_t = df_kp.index.max()
    else:
        # Fallback to last 7 days
        end_t = datetime.now(tz=timedelta(0))
        start_t = end_t - timedelta(days=7)
        
    minute_index = pd.date_range(start=start_t, end=end_t, freq='1min', tz='UTC')
    df_sw = pd.DataFrame(index=minute_index)
    
    if not df_kp.empty:
        df_sw = df_sw.join(df_kp)
    if not df_xray_pivot.empty:
        df_sw = df_sw.join(df_xray_pivot)
        
    # Forward fill 1-minute gaps up to 5 minutes
    df_sw = df_sw.ffill(limit=5)
    
    # Forward fill daily F10.7 flux
    if not df_f107.empty:
        # Resample to 1 minute and merge
        df_f107_1m = df_f107.resample('1min').ffill()
        df_sw = df_sw.join(df_f107_1m, how='left')
        df_sw['F107'] = df_sw['F107'].ffill()
    else:
        # Standard solar flux default if API fails
        df_sw['F107'] = 150.0
        
    # Map 'kp_index' to 'Kp_index' and 'estimated_kp'
    if 'kp_index' in df_sw.columns:
        if 'estimated_kp' in df_sw.columns:
            df_sw['Kp_index'] = df_sw['kp_index'].fillna(df_sw['estimated_kp'])
        else:
            df_sw['Kp_index'] = df_sw['kp_index']
    elif 'estimated_kp' in df_sw.columns:
        df_sw['Kp_index'] = df_sw['estimated_kp']
    else:
        df_sw['Kp_index'] = 2.0  # Default fallback
        
    df_sw['Kp_index'] = df_sw['Kp_index'].fillna(2.0)
    
    # Fill in Ap index from Kp index using standard conversions if Ap not present
    kp_to_ap = {
        0.0: 0, 0.3: 2, 0.7: 3, 1.0: 4, 1.3: 5, 1.7: 6, 2.0: 7, 2.3: 9, 2.7: 12,
        3.0: 15, 3.3: 18, 3.7: 22, 4.0: 27, 4.3: 32, 4.7: 39, 5.0: 48, 5.3: 56,
        5.7: 67, 6.0: 80, 6.3: 94, 6.7: 111, 7.0: 132, 7.3: 154, 7.7: 179,
        8.0: 207, 8.3: 236, 8.7: 300, 9.0: 400
    }
    
    def get_ap(kp):
        if pd.isna(kp):
            return 7
        closest_kp = min(kp_to_ap.keys(), key=lambda x: abs(x - kp))
        return kp_to_ap[closest_kp]
        
    df_sw['Ap'] = df_sw['Kp_index'].apply(get_ap)
    
    # Ensure all required columns exist
    for col in ['Xray_flux_short', 'Xray_flux_long']:
        if col not in df_sw.columns:
            df_sw[col] = 1e-7 # reasonable quiet background flux
            
    # Save to compiled recent file
    recent_path = os.path.join(base_dir, 'compiled_recent_1m.csv')
    df_sw.index.name = 'Datetime'
    df_sw.to_csv(recent_path)
    print(f"Live Space Weather compiled. Saved {len(df_sw)} records to {recent_path}")

def fetch_tle_data():
    base_dir = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
    out_file = os.path.join(base_dir, 'data', 'starlink_history.csv')
    
    username = os.environ.get('SPACETRACK_USERNAME')
    password = os.environ.get('SPACETRACK_PASSWORD')
    
    if username and password:
        print("Authenticating with Space-Track to get live TLE data...")
        session = requests.Session()
        try:
            login_url = "https://www.space-track.org/ajaxauth/login"
            login_data = {'identity': username, 'password': password}
            resp = session.post(login_url, data=login_data, timeout=30)
            if resp.status_code == 200:
                print("Authenticated successfully. Fetching recent Starlink TLEs...")
                query_url = "https://www.space-track.org/basicspacedata/query/class/gp/OBJECT_NAME/~~STARLINK/EPOCH/>now-7/format/csv"
                res = session.get(query_url, timeout=120)
                res.raise_for_status()
                
                text = res.text.strip()
                if text and "No data found" not in text and "NO RESULTS RETURNED" not in text:
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(text + '\n')
                    print(f"Successfully downloaded live Starlink TLEs to {out_file}")
                    return
                else:
                    print("No live Starlink data returned from query. Falling back to local data...")
            else:
                print(f"Login failed! Status code: {resp.status_code}. Falling back to local data...")
        except Exception as e:
            print(f"Error fetching live Space-Track data: {e}. Falling back to local data...")
            
    # Local fallback
    print("Using local TLE data files as fallback...")
    gp_path = os.path.join(base_dir, 'gp.txt')
    data123_path = os.path.join(base_dir, 'data123.txt')
    
    records = []
    
    def parse_3line_file(filepath):
        parsed = []
        if not os.path.exists(filepath):
            return parsed
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        for idx in range(0, len(lines) - 2, 3):
            name = lines[idx]
            line1 = lines[idx+1]
            line2 = lines[idx+2]
            
            if not (line1.startswith('1 ') and line2.startswith('2 ')):
                # If there are comment lines or format mismatches, step single lines forward
                continue
                
            norad_id = int(line1[2:7])
            epoch_str = line1[18:32]
            yy = int(epoch_str[0:2])
            ddd = float(epoch_str[2:])
            year = 2000 + yy if yy < 57 else 1900 + yy
            epoch_date = datetime(year, 1, 1) + timedelta(days=ddd - 1)
            epoch_iso = epoch_date.strftime('%Y-%m-%dT%H:%M:%S.%f')
            
            parsed.append({
                'OBJECT_NAME': name,
                'NORAD_CAT_ID': norad_id,
                'EPOCH': epoch_iso,
                'TLE_LINE1': line1,
                'TLE_LINE2': line2
            })
        return parsed

    print("Parsing gp.txt...")
    records.extend(parse_3line_file(gp_path))
    print(f"Parsed {len(records)} records from gp.txt")
    
    print("Parsing data123.txt...")
    new_records = parse_3line_file(data123_path)
    records.extend(new_records)
    print(f"Parsed {len(new_records)} records from data123.txt")
    
    if len(records) > 0:
        df_tle = pd.DataFrame(records)
        df_tle.to_csv(out_file, index=False)
        print(f"Compiled TLE records into {out_file} (Rows: {len(df_tle)})")
    else:
        print("No local TLE files found! Checking for CSV element files...")
        # Check if datataw.txt is present
        datataw_path = os.path.join(base_dir, 'datataw.txt')
        if os.path.exists(datataw_path):
            try:
                # datataw.txt has CSV structure but might lack TLE lines.
                # However, for training we need TLE lines to pass to SGP4 (Satrec.twoline2rv)
                # If we only have datataw.txt, we can generate TLE lines or read them.
                # Since gp.txt and data123.txt cover all Starlinks, they will be the primary source.
                print("datataw.txt is present but gp.txt and data123.txt are preferred since they contain actual TLE lines.")
            except Exception as e:
                print(f"Error checking datataw.txt: {e}")

def main():
    compile_live_space_weather()
    fetch_tle_data()

if __name__ == "__main__":
    main()
