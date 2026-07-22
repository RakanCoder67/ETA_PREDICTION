"""
generate_real_comparison_graphs.py
------------------------------------
Generates honest comparison_24h.png and comparison_30d.png directly
from the real evaluation_dataset.csv — no artificial capping, no formulas.
Uses real sgp4_error_3d and ml_error_3d from validated TLE pair evaluations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
CSV_PATH = os.path.join(BASE_DIR, "models", "evaluation_dataset.csv")
OUT_DIR  = os.path.join(BASE_DIR, "models")

C_SGP4 = "#e63946"
C_ML   = "#06d6a0"
C_HPOP = "#ff9f1c"

print("Loading real evaluation dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df):,} real TLE pair records loaded.")

# Use real 3D errors directly
df["sgp4_error_3d"] = pd.to_numeric(df["sgp4_error_3d"], errors="coerce")
df["ml_error_3d"]   = pd.to_numeric(df["ml_error_3d"],   errors="coerce")
df["hpop_error_3d"] = pd.to_numeric(df["hpop_error_3d"], errors="coerce")
df["dt_hours"]      = pd.to_numeric(df["dt_hours"],       errors="coerce")
df = df.dropna(subset=["sgp4_error_3d", "ml_error_3d", "dt_hours"])

# Filter out extreme outliers (>2000 km — corrupted TLE entries)
df = df[(df["sgp4_error_3d"] < 2000) & (df["ml_error_3d"] < 2000)]
print(f"  {len(df):,} records after outlier filter.")

def make_chart(df_sub, window_label, filename, title_suffix):
    # Group into 1h bins and aggregate as median (robust to tail outliers)
    df_sub = df_sub.copy()
    df_sub["dt_bin"] = df_sub["dt_hours"].round(0).astype(int)
    
    grp = df_sub.groupby("dt_bin").agg(
        sgp4=("sgp4_error_3d", "median"),
        ml=("ml_error_3d",   "median"),
        hpop=("hpop_error_3d","median"),
        n=("sgp4_error_3d",  "count")
    ).reset_index()
    
    # Only keep bins with at least 3 samples
    grp = grp[grp["n"] >= 3].sort_values("dt_bin")
    
    t  = grp["dt_bin"].values
    s  = grp["sgp4"].values
    m  = grp["ml"].values
    h  = grp["hpop"].values
    
    improvement_pct = (s - m) / s * 100
    mean_improvement = improvement_pct.mean()
    
    print(f"\n  {window_label} — {len(grp)} time bins, {df_sub['n'].sum() if 'n' in df_sub else len(df_sub):,} samples")
    print(f"  Median SGP4 error: {np.median(s):.2f} km")
    print(f"  Median ML error:   {np.median(m):.2f} km")
    print(f"  Mean improvement:  {mean_improvement:.1f}%")
    
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    ax.fill_between(t, s, alpha=0.08, color=C_SGP4)
    ax.fill_between(t, m, alpha=0.12, color=C_ML)
    
    ax.plot(t, s, color=C_SGP4, linewidth=2.2, label=f"Standard SGP4 (median {np.median(s):.1f} km)")
    ax.plot(t, h, color=C_HPOP, linewidth=1.8, linestyle="--", label=f"HPOP Benchmark (median {np.median(h):.1f} km)", alpha=0.85)
    ax.plot(t, m, color=C_ML,   linewidth=2.5, label=f"ML Corrector — Ours (median {np.median(m):.2f} km)", zorder=5)
    
    # ETA annotation
    ax.axhline(y=np.median(m), color=C_ML, linestyle=":", linewidth=1.2, alpha=0.5)
    
    ax.set_xlabel("Time After TLE Epoch (hours)", fontsize=12, color="black")
    ax.set_ylabel("3D Position Error (km)", fontsize=12, color="black")
    ax.set_title(f"Real Prediction Error vs Time — {title_suffix}\n"
                 f"ML model reduces error by {mean_improvement:.1f}% vs SGP4  |  N = {len(df_sub):,} real TLE pairs",
                 fontsize=13, color="black", pad=12)
    
    ax.tick_params(colors="black", labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.grid(True, color="#e5e5e5", linestyle=":", linewidth=0.8)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    
    legend = ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc", loc="upper left")
    for text in legend.get_texts():
        text.set_color("black")
    
    out = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# 24-hour chart
df_24h = df[df["dt_hours"] <= 24.5]
make_chart(df_24h, "24-Hour", "comparison_24h.png", "24-Hour Horizon")

# 30-day chart
df_30d = df[df["dt_hours"] <= 720]
make_chart(df_30d, "30-Day", "comparison_30d.png", "30-Day Horizon")

print("\nDone. Both charts generated from real evaluation data.")
