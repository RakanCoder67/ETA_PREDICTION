"""
fast_sample_tle.py
------------------
Reads the 2.7 GB starlink_history.csv in streaming chunks and saves a
sampled subset (up to --sats unique satellites) to data/starlink_sample.csv.

This avoids loading the full file into RAM.

Usage:
    python fast_sample_tle.py          # 300 satellites (default)
    python fast_sample_tle.py --sats 500
"""

import os, argparse
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
IN_CSV   = os.path.join(BASE_DIR, "data", "starlink_history.csv")
OUT_CSV  = os.path.join(BASE_DIR, "data", "starlink_sample.csv")

KEEP_COLS = ["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH", "TLE_LINE1", "TLE_LINE2"]
CHUNK     = 50_000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Streaming {IN_CSV} in {CHUNK:,}-row chunks ...")

    # Pass 1: collect all unique NORAD IDs (only read 1 column)
    print("Pass 1: scanning NORAD IDs ...", flush=True)
    all_ids = set()
    for chunk in pd.read_csv(IN_CSV, usecols=["NORAD_CAT_ID"],
                              chunksize=CHUNK, on_bad_lines="skip"):
        all_ids.update(chunk["NORAD_CAT_ID"].dropna().astype(int).tolist())
    print(f"  Found {len(all_ids):,} unique satellites.")

    # Sample target IDs
    rng    = np.random.default_rng(args.seed)
    n_pick = min(args.sats, len(all_ids))
    picked = set(int(x) for x in rng.choice(list(all_ids), n_pick, replace=False))
    print(f"  Selected {len(picked):,} satellites to extract.")

    # Pass 2: stream and keep only picked rows
    print("Pass 2: extracting rows ...", flush=True)
    parts, rows = [], 0
    for i, chunk in enumerate(pd.read_csv(IN_CSV, usecols=KEEP_COLS,
                                            chunksize=CHUNK, on_bad_lines="skip")):
        sub = chunk[chunk["NORAD_CAT_ID"].isin(picked)]
        if len(sub):
            parts.append(sub)
            rows += len(sub)
        if (i + 1) % 20 == 0:
            print(f"  scanned {(i+1)*CHUNK:,} rows | kept {rows:,}", flush=True)

    result = pd.concat(parts, ignore_index=True)
    result.to_csv(OUT_CSV, index=False)
    n_sats = result["NORAD_CAT_ID"].nunique()
    print(f"\nSaved {len(result):,} rows ({n_sats:,} satellites) -> {OUT_CSV}")

if __name__ == "__main__":
    main()
