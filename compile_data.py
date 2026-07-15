import pandas as pd
import json
import os
import numpy as np

def load_kp_ap_since_1932(filepath):
    # Fixed length or blank separated. Let's use read_csv with delim_whitespace
    df = pd.read_csv(filepath, comment='#', sep=r'\s+', header=None)
    df.columns = ['YYYY', 'MM', 'DD', 'days', 'days_m', 'Bsr', 'dB',
                  'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8',
                  'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8',
                  'Ap', 'SN', 'F10.7obs', 'F10.7adj', 'D']
    
    df['Date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
    df.replace([-1.000, -1, -1.0], np.nan, inplace=True)
    df.set_index('Date', inplace=True)
    # Drop original date columns to keep it clean
    df.drop(columns=['YYYY', 'MM', 'DD'], inplace=True)
    return df

def load_sunspot_number(filepath):
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['Year', 'Month', 'Day', 'DecimalYear', 'SN', 'StdDev', 'Observations', 'Marker'])
    # df.columns = ['Year', 'Month', 'Day', 'DecimalYear', 'SN', 'StdDev', 'Observations', 'Marker']
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
    df['SN'] = df['SN'].replace(-1, np.nan)
    df.set_index('Date', inplace=True)
    df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
    return df

def load_fluxtable(filepath):
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=[1])
    # Flux table dates are like 20041028 and times like 170000
    # Make sure times are padded with 0 to 6 chars
    times = df['fluxtime'].astype(str).str.zfill(6)
    df['Datetime'] = pd.to_datetime(df['fluxdate'].astype(str) + times, format='%Y%m%d%H%M%S', errors='coerce')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['fluxdate', 'fluxtime'], inplace=True)
    return df

def load_planetary_k_index(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['Datetime'] = pd.to_datetime(df['time_tag'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['time_tag'], inplace=True)
    return df

def load_xrays(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['Datetime'] = pd.to_datetime(df['time_tag'], utc=True)
    
    # Pivot by energy to have short and long bands as separate columns
    # In some rare cases there might be duplicate timestamps, we use 'mean' to aggregate
    df_pivot = df.pivot_table(index='Datetime', columns='energy', values='flux', aggfunc='mean')
    df_pivot.rename(columns={'0.05-0.4nm': 'Xray_flux_short', '0.1-0.8nm': 'Xray_flux_long'}, inplace=True)
    return df_pivot

def main():
    base_dir = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
    
    print("Loading datasets...")
    df_kp = load_kp_ap_since_1932(os.path.join(base_dir, 'Kp_ap_Ap_SN_F107_since_1932.txt'))
    df_sn = load_sunspot_number(os.path.join(base_dir, 'SN_d_tot_V2.0.txt'))
    df_flux = load_fluxtable(os.path.join(base_dir, 'fluxtable.txt'))
    
    df_k_1m = load_planetary_k_index(os.path.join(base_dir, 'planetary_k_index_1m.json'))
    df_xray_1m = load_xrays(os.path.join(base_dir, 'xrays-7-day.json'))
    
    print("Exporting Historical Dataset (Daily resolution)...")
    # We will export the comprehensive Kp file as our primary historical set
    # It already contains daily Ap, SN, and F10.7 which are most commonly used
    df_historical = df_kp.copy()
    
    # We can also add fluxtable (by aggregating to daily mean) to add highly accurate flux data
    # Need to numeric conversion in case of issues
    df_flux_numeric = df_flux.apply(pd.to_numeric, errors='coerce')
    df_flux_daily = df_flux_numeric.resample('D').mean()
    df_historical = df_historical.join(df_flux_daily[['fluxobsflux', 'fluxadjflux', 'fluxursi']], rsuffix='_penticton')
    
    historical_path = os.path.join(base_dir, 'compiled_historical_daily.csv')
    df_historical.to_csv(historical_path)
    print(f"- {historical_path} created (Rows: {len(df_historical)})")
    
    print("Exporting Recent 1-minute Dataset...")
    # Join 1-minute JSON datasets
    # Align to a continuous 1-minute grid
    recent_start = min(df_k_1m.index.min(), df_xray_1m.index.min())
    recent_end = max(df_k_1m.index.max(), df_xray_1m.index.max())
    minute_index = pd.date_range(start=recent_start, end=recent_end, freq='1min', tz='UTC')
    
    df_recent = pd.DataFrame(index=minute_index)
    df_recent = df_recent.join(df_k_1m)
    df_recent = df_recent.join(df_xray_1m)
    
    # Forward fill small gaps (e.g., up to 5 minutes missing)
    df_recent = df_recent.ffill(limit=5)
    
    # Join daily variables from historical for these recent days using forward fill
    # Convert historical Date index (tz-naive) to UTC for joining
    df_historical_tz = df_historical.copy()
    df_historical_tz.index = df_historical_tz.index.tz_localize('UTC')
    
    # Forward fill daily data to the minute-by-minute index
    df_recent = df_recent.join(df_historical_tz[['Ap', 'SN', 'F10.7obs', 'F10.7adj', 'fluxobsflux']], how='left')
    df_recent[['Ap', 'SN', 'F10.7obs', 'F10.7adj', 'fluxobsflux']] = df_recent[['Ap', 'SN', 'F10.7obs', 'F10.7adj', 'fluxobsflux']].ffill()
    
    recent_path = os.path.join(base_dir, 'compiled_recent_1m.csv')
    df_recent.to_csv(recent_path)
    print(f"- {recent_path} created (Rows: {len(df_recent)})")
    
    print("\nData compilation complete! You can now use these CSVs in your ETA Prediction program.")

if __name__ == "__main__":
    main()
