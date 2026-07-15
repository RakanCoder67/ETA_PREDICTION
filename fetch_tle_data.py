"""
fetch_tle_data.py
-----------------
Downloads Starlink TLE history from Space-Track.org in 7-day chunks.

Modes:
  Default (no flags): resume from the latest epoch in the existing file.
  --full             : re-download the last LOOKBACK_DAYS (ignores existing file).
  --backfill         : download data OLDER than the earliest epoch in the file,
                       appending historical data behind what we already have.

Credentials are read from environment variables:
    set SPACETRACK_USERNAME=your_email@example.com
    set SPACETRACK_PASSWORD=yourpassword
"""

import os
import argparse
import requests
import time
import pandas as pd
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OUT_FILE = 'data/starlink_history.csv'
LOOKBACK_DAYS = 540          # How far back to go on a fresh download (18 months)
CHUNK_DAYS = 3               # Smaller API-safe chunk size to prevent Space-Track 500 errors
SLEEP_BETWEEN_CHUNKS = 4     # seconds — respects Space-Track limits
REQUEST_TIMEOUT = 180        # seconds per API call


def get_epoch_range(out_file: str):
    """
    Read the existing CSV and return (earliest_epoch, latest_epoch) as naive datetimes.
    Returns (None, None) if file doesn't exist or is empty.
    """
    if not os.path.exists(out_file):
        return None, None
    try:
        df = pd.read_csv(out_file, usecols=['EPOCH'])
        if df.empty:
            return None, None
        epochs = pd.to_datetime(df['EPOCH'])
        return epochs.min().replace(tzinfo=None), epochs.max().replace(tzinfo=None)
    except Exception as e:
        print(f"Warning: could not read existing file ({e}).")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Download Starlink TLE history from Space-Track")
    parser.add_argument('--full',     action='store_true',
                        help=f'Re-download the last {LOOKBACK_DAYS} days (overwrites existing file)')
    parser.add_argument('--backfill', action='store_true',
                        help='Download data older than the earliest epoch in the existing file')
    args = parser.parse_args()

    username = os.environ.get('SPACETRACK_USERNAME')
    password = os.environ.get('SPACETRACK_PASSWORD')

    if not username or not password:
        print("Error: SPACETRACK_USERNAME and/or SPACETRACK_PASSWORD are not set.")
        print("Set them in PowerShell with:")
        print("  $env:SPACETRACK_USERNAME = 'your_email@example.com'")
        print("  $env:SPACETRACK_PASSWORD = 'yourpassword'")
        return

    # ── Determine date window ──
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    earliest_in_file, latest_in_file = get_epoch_range(OUT_FILE)

    if args.full or earliest_in_file is None:
        # Full re-download: last LOOKBACK_DAYS up to today
        start_date = today - timedelta(days=LOOKBACK_DAYS)
        end_date   = today
        append_mode = False
        print(f"Full download: {start_date.date()} to {end_date.date()} "
              f"({LOOKBACK_DAYS} days of Starlink TLE history)")
    elif args.backfill:
        # Download historical data BEFORE what we already have
        end_date   = earliest_in_file.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - timedelta(days=LOOKBACK_DAYS)
        if start_date >= end_date:
            print(f"Nothing to backfill: file already covers back to {earliest_in_file.date()}.")
            return
        append_mode = True
        print(f"Backfill mode: {start_date.date()} to {end_date.date()} "
              f"(adding historical data before {end_date.date()})")
    else:
        # Resume: download data AFTER the latest epoch we have
        start_date = (latest_in_file + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        end_date   = today
        if start_date >= end_date:
            print(f"Database is already up to date (latest epoch: {latest_in_file.date()}). "
                  "Nothing to download.")
            print("Tip: run with --backfill to download older historical data.")
            return
        append_mode = True
        print(f"Resume mode: continuing from {start_date.date()} to {end_date.date()}")

    # ── Authenticate ──
    session = requests.Session()
    print("Authenticating to Space-Track.org...")
    login_resp = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={'identity': username, 'password': password},
        timeout=30
    )
    if login_resp.status_code != 200 or 'Failed' in login_resp.text:
        print(f"Login failed! HTTP {login_resp.status_code}")
        print(login_resp.text[:300])
        return
    print("Authentication successful.\n")

    # ── Setup output file ──
    os.makedirs('data', exist_ok=True)
    file_mode = 'a' if append_mode else 'w'
    first_write = not append_mode   # only write CSV header on a fresh file

    total_rows = 0
    current_start = start_date

    print(f"Downloading in {CHUNK_DAYS}-day chunks (3-second pause between each)...")
    print(f"Output: {OUT_FILE}\n")

    with open(OUT_FILE, file_mode, encoding='utf-8') as f:
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)

            start_str = current_start.strftime('%Y-%m-%d')
            end_str   = current_end.strftime('%Y-%m-%d')

            query_url = (
                "https://www.space-track.org/basicspacedata/query"
                "/class/gp_history/OBJECT_NAME/~~STARLINK"
                f"/EPOCH/{start_str}--{end_str}"
                "/format/csv/emptyresult/show"
            )

            try:
                res = session.get(query_url, timeout=REQUEST_TIMEOUT)
                res.raise_for_status()
                text = res.text.strip()

                if text and "No data found" not in text:
                    lines = text.splitlines()
                    n_rows = max(0, len(lines) - 1)   # exclude header

                    if first_write:
                        f.write('\n'.join(lines) + '\n')
                        first_write = False
                    else:
                        if len(lines) > 1:
                            f.write('\n'.join(lines[1:]) + '\n')

                    total_rows += n_rows
                    print(f"  {start_str} to {end_str} : {n_rows:>6,} rows  "
                          f"(total so far: {total_rows:,})")
                else:
                    print(f"  {start_str} to {end_str} : no data")

            except requests.exceptions.RequestException as e:
                print(f"  {start_str} to {end_str} : REQUEST ERROR - {e}")
                print("  Retrying once after 10s...")
                time.sleep(10)
                try:
                    res = session.get(query_url, timeout=REQUEST_TIMEOUT)
                    res.raise_for_status()
                    text = res.text.strip()
                    if text and "No data found" not in text:
                        lines = text.splitlines()
                        n_rows = max(0, len(lines) - 1)
                        if first_write:
                            f.write('\n'.join(lines) + '\n')
                            first_write = False
                        else:
                            if len(lines) > 1:
                                f.write('\n'.join(lines[1:]) + '\n')
                        total_rows += n_rows
                        print(f"  Retry OK: {n_rows:,} rows")
                except Exception as e2:
                    print(f"  Retry also failed: {e2}. Skipping this chunk.")

            current_start = current_end
            if current_start < end_date:
                time.sleep(SLEEP_BETWEEN_CHUNKS)

    print(f"\nDownload complete!")
    print(f"Total rows written this session: {total_rows:,}")
    print(f"File saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
