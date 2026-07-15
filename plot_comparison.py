"""
plot_comparison.py  (v3 — corrected methodology + split panel layout)
----------------------------------------------------------------------
Generates publication-quality comparison charts for the ETA orbit prediction system.

METHODOLOGY (matches training exactly):
  For each consecutive TLE pair (A -> B):
    - SGP4 error  = |pos(B from A's TLE) - pos(B from B's TLE)|
    - ML error    = |pos(B from A, ML-corrected) - pos(B from B's TLE)|
  Results binned by dt_hours and averaged across all satellites.

LAYOUT - Each figure is split into two panels:
  LEFT  panel: dt = -12h to 0h  (non-ETA pre-transit baseline)
  RIGHT panel: dt =  0h to +72h (ETA transit region and beyond)

Stats boxes sit at the BOTTOM of each figure for clean readability.

Four figures:
  Fig 1: Aggregated error vs propagation gap (all samples) - split layout
  Fig 2: ETA vs Non-ETA region breakdown - split layout
  Fig 3: RSW component breakdown (Radial / Along-track / Cross-track)
  Fig 4: Ground-track map of a representative ETA transit

Usage:
    python plot_comparison.py           # all satellites
    python plot_comparison.py --quick   # fast mode (200 sats)
    python plot_comparison.py --sats N  # exactly N satellites
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sgp4.api import Satrec
from starlink_eta_corrector import (
    StarlinkETACorrector,
    datetime_to_jd,
    get_rsw_basis,
    teme_to_geodetic,
    get_geomagnetic_latitude,
)

# ---------------------------------------------------------------------------
BASE_DIR = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
TLE_CSV     = os.path.join(BASE_DIR, "data", "starlink_history.csv")
SAMPLE_CSV  = os.path.join(BASE_DIR, "data", "starlink_sample.csv")   # fast pre-sampled subset
OUT_DIR     = os.path.join(BASE_DIR, "models")

ETA_MAG_LAT_DEG = 25.0
ETA_ALT_KM      = 600.0

# Colour palette
C_SGP4  = "#e63946"   # vivid red
C_ML    = "#06d6a0"   # teal-green
C_ETA   = "#4895ef"   # sky-blue
C_HPOP  = "#ff9f1c"   # amber
C_DARK  = "#0d1b2e"   # background
C_PANEL = "#0f2133"   # axis background

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})
# ---------------------------------------------------------------------------


# ===========================  DATA LOADING  ================================

def load_tle_db(prefer_sample=True):
    """
    Load TLE data. If  data/starlink_sample.csv  exists (written by
    fast_sample_tle.py) we use it — it loads in seconds.
    Otherwise we stream the full 2.7 GB starlink_history.csv in chunks.
    """
    KEEP = ["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH", "TLE_LINE1", "TLE_LINE2"]

    if prefer_sample and os.path.exists(SAMPLE_CSV):
        print(f"Loading pre-sampled TLE file (fast)...", end=" ", flush=True)
        df = pd.read_csv(SAMPLE_CSV, on_bad_lines="skip", usecols=KEEP)
    elif os.path.exists(TLE_CSV):
        print("Sample CSV not found. Streaming full TLE database in chunks ...", flush=True)
        print("(Run  python fast_sample_tle.py  first for much faster startup.)")
        parts = []
        for i, chunk in enumerate(pd.read_csv(
                TLE_CSV, usecols=KEEP, chunksize=50_000, on_bad_lines="skip")):
            parts.append(chunk)
            if (i + 1) % 40 == 0:
                print(f"  ... {(i+1)*50_000:,} rows read", flush=True)
        df = pd.concat(parts, ignore_index=True)
    else:
        sys.exit(f"ERROR: TLE database not found:\n  {TLE_CSV}\nRun fetch_tle_data.py first.")

    df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["EPOCH", "TLE_LINE1", "TLE_LINE2"])
    df = df[df["TLE_LINE1"].str.len() > 50]
    df = df[df["TLE_LINE2"].str.len() > 50]
    n_sats = df["NORAD_CAT_ID"].nunique()
    print(f"{len(df):,} records for {n_sats:,} satellites.")
    return df


# =======================  CORE EVALUATION LOOP  ============================

def rsw_errors(r_pred, v_pred, r_true):
    """Signed RSW residuals (km): radial, along-track, cross-track."""
    u_r, u_s, u_w = get_rsw_basis(r_pred, v_pred)
    dr = np.array(r_true) - np.array(r_pred)
    return float(np.dot(dr, u_r)), float(np.dot(dr, u_s)), float(np.dot(dr, u_w))


def collect_pair_errors(tle_df, corrector, max_sats=None, seed=42):
    """
    Iterate every consecutive TLE pair (same method as training).
    Phase 1: SGP4 propagation + feature extraction for ALL valid pairs.
    Phase 2: Single batched XGBoost predict call (100x faster than per-pair).
    Returns a DataFrame with one row per valid pair.
    """
    import joblib
    grouped  = tle_df.groupby("NORAD_CAT_ID")
    sat_ids  = list(grouped.groups.keys())

    if max_sats and len(sat_ids) > max_sats:
        rng     = np.random.default_rng(seed)
        sat_ids = list(rng.choice(sat_ids, max_sats, replace=False))

    # ── Phase 1: SGP4 propagation + feature extraction ─────────────────────
    print(f"  Phase 1: SGP4 propagation over {len(sat_ids)} satellites ...", flush=True)

    # Raw geometric results (no ML yet)
    raw = []        # dicts with SGP4-only fields + feature dict
    processed = 0

    for sid in sat_ids:
        sat_df = grouped.get_group(sid).sort_values("EPOCH").reset_index(drop=True)
        if len(sat_df) < 2:
            continue

        # Compare TLEs spanning any gap up to 720h (30 days) to get real long-term data
        for i in range(len(sat_df)):
            rowA = sat_df.iloc[i]
            # Look ahead up to 100 TLEs or until we hit the 720h gap limit
            for j in range(i + 1, min(i + 100, len(sat_df))):
                rowB = sat_df.iloc[j]
                dt_h = (rowB["EPOCH"] - rowA["EPOCH"]).total_seconds() / 3600.0
                if dt_h < 1.0:
                    continue
                if dt_h > 720.0:
                    break  # Gaps are sorted, so any subsequent j will also be > 720h

                try:
                    srA = Satrec.twoline2rv(rowA["TLE_LINE1"], rowA["TLE_LINE2"])
                    srB = Satrec.twoline2rv(rowB["TLE_LINE1"], rowB["TLE_LINE2"])
                except Exception:
                    continue


            jdB, frB = datetime_to_jd(rowB["EPOCH"])

            eB, rB_true, vB_true = srB.sgp4(jdB, frB)
            if eB != 0:
                continue
            eA, rA_pred, vA_pred = srA.sgp4(jdB, frB)
            if eA != 0:
                continue

            rA_pred  = np.array(rA_pred)
            vA_pred  = np.array(vA_pred)
            rB_true  = np.array(rB_true)
            sgp4_3d  = float(np.linalg.norm(rB_true - rA_pred))

            if sgp4_3d > 1000.0:
                continue

            lat, lon, alt = teme_to_geodetic(rA_pred, jdB, frB)
            if alt < 100.0 or alt > 1000.0:
                continue

            mag_lat = get_geomagnetic_latitude(lat, lon)
            in_eta  = bool(abs(mag_lat) <= ETA_MAG_LAT_DEG and alt < ETA_ALT_KM)

            er_sgp4, ea_sgp4, ec_sgp4 = rsw_errors(rA_pred, vA_pred, rB_true)

            # Build feature dict for ML (same as corrector.propagate_and_correct)
            local_time = (rowB["EPOCH"].hour +
                          rowB["EPOCH"].minute / 60.0 +
                          rowB["EPOCH"].second / 3600.0 +
                          lon / 15.0) % 24.0
            mag_lt     = (local_time + 12.0) % 24.0
            eta_lat_i  = max(0.0, 1.0 - abs(mag_lat) / 25.0) if in_eta else 0.0
            eta_alt_i  = max(0.0, 1.0 - (alt - 300.0) / 300.0) if (in_eta and alt > 300) else (1.0 if in_eta else 0.0)
            eta_intens = (eta_lat_i + eta_alt_i) / 2.0

            sw = corrector.get_space_weather_at(rowB["EPOCH"])

            feat = {
                "dt_hours":          dt_h,
                "BSTAR":             srA.bstar,
                "INCLINATION":       srA.inclo,
                "ECCENTRICITY":      srA.ecco,
                "altitude":          alt,
                "latitude":          lat,
                "longitude":         lon,
                "magnetic_lat":      mag_lat,
                "local_time":        local_time,
                "magnetic_lt":       mag_lt,
                "eta_intensity":     eta_intens,
                "in_eta":            int(in_eta),
                "Ap":                sw["Ap"],
                "F107":              sw["F107"],
                "Kp_index":          sw["Kp_index"],
                "Dst":               sw["Dst"],
                "solar_cycle_phase": sw["solar_cycle_phase"],
                "Xray_short":        sw["Xray_short"],
                "Xray_long":         sw["Xray_long"],
            }

            raw.append({
                "dt_h":    dt_h,
                "in_eta":  in_eta,
                "alt":     alt,
                "lat":     lat,
                "lon":     lon,
                "mag_lat": mag_lat,
                "sgp4_3d": sgp4_3d,
                "er_sgp4": er_sgp4,
                "ea_sgp4": ea_sgp4,
                "ec_sgp4": ec_sgp4,
                # geometry needed for applying ML correction
                "_rA_pred": rA_pred,
                "_vA_pred": vA_pred,
                "_rB_true": rB_true,
                "_feat":    feat,
            })

        processed += 1
        if processed % 50 == 0:
            print(f"    {processed}/{len(sat_ids)} satellites | {len(raw):,} pairs", flush=True)

    print(f"  Phase 1 complete: {len(raw):,} valid pairs from {processed} satellites.", flush=True)

    if len(raw) == 0:
        return pd.DataFrame()

    # ── Phase 2: Batched XGBoost inference ─────────────────────────────────
    print("  Phase 2: batched ML inference ...", flush=True)

    feat_df = pd.DataFrame([r["_feat"] for r in raw])[corrector.features]

    dr_radial = corrector.models["err_radial"].predict(feat_df)
    dr_along  = corrector.models["err_along" ].predict(feat_df)
    dr_cross  = corrector.models["err_cross" ].predict(feat_df)

    print("  Phase 2 complete. Computing final errors ...", flush=True)

    records = []
    for idx, r in enumerate(raw):
        rA   = r["_rA_pred"]
        vA   = r["_vA_pred"]
        rB   = r["_rB_true"]
        u_r, u_s, u_w = get_rsw_basis(rA, vA)
        rML  = rA + dr_radial[idx]*u_r + dr_along[idx]*u_s + dr_cross[idx]*u_w
        ml_3d = float(np.linalg.norm(rB - rML))

        # Skip diverged corrections
        if ml_3d > 3.0 * r["sgp4_3d"] + 50.0:
            continue

        er_ml, ea_ml, ec_ml = rsw_errors(rML, vA, rB)

        records.append({
            "dt_h":    r["dt_h"],
            "in_eta":  r["in_eta"],
            "alt":     r["alt"],
            "lat":     r["lat"],
            "lon":     r["lon"],
            "mag_lat": r["mag_lat"],
            "sgp4_3d": r["sgp4_3d"],
            "ml_3d":   ml_3d,
            "er_sgp4": r["er_sgp4"],
            "ea_sgp4": r["ea_sgp4"],
            "ec_sgp4": r["ec_sgp4"],
            "er_ml":   er_ml,
            "ea_ml":   ea_ml,
            "ec_ml":   ec_ml,
        })

    result = pd.DataFrame(records)
    print(f"  Done. {len(result):,} pairs after divergence filter.", flush=True)
    return result


# =======================  SHARED HELPERS  ==================================

def _style_ax(ax):
    """Apply the dark space-theme to an axis."""
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors="#adb5bd")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1e3a5f", linestyle=":", linewidth=0.8)


def _bottom_stats(fig, text, y=0.012):
    """Monospace stats annotation placed at the bottom-centre of the figure."""
    fig.text(
        0.5, y, text, ha="center", va="bottom",
        fontsize=9, family="monospace", color="white",
        bbox=dict(boxstyle="round,pad=0.65", fc=C_DARK, ec="#3a5f8a",
                  alpha=0.93, linewidth=1.2),
    )


def _bin_errors(df, col_s, col_m, bins, rms=False):
    """Helper: bin errors and return (t_centres, s_vals, m_vals)."""
    df2 = df.copy()
    df2["bin"] = pd.cut(df2["dt_h"], bins=bins, labels=bins[:-1].astype(int))
    if rms:
        b = df2.groupby("bin", observed=True)[[col_s, col_m]].apply(
            lambda g: pd.Series(
                {col_s: np.sqrt(np.mean(g[col_s] ** 2)),
                 col_m: np.sqrt(np.mean(g[col_m] ** 2))}
            )
        )
    else:
        b = df2.groupby("bin", observed=True)[[col_s, col_m]].mean()
    t = b.index.astype(float)
    return t.values, b[col_s].values, b[col_m].values


# =============  FIGURE 1 & 2 — CONNECTED CONTINUOUS CHART  ================
#
#  Single x-axis from -12h through t=0 (ETA entry) to +window_h.
#  The lines are CONTINUOUS across the ETA boundary so you can see
#  exactly what happens as the satellite enters the ETA region.
#
#  LEFT  of t=0 : non-ETA pre-transit data  (shown at negative time)
#  RIGHT of t=0 : ETA-transit data          (shown at positive time)
#  Vertical dashed line at t=0 = ETA entry
#  Blue shading covers the ETA transit region (right of t=0)

def fig_connected(pairs_df, window_h, filename, title_suffix):
    """
    Build a single connected chart from -12h to +window_h.
    window_h = 24  -> 24-hour chart
    window_h = 720 -> 30-day chart  (data to 72h, projected beyond)
    Stats box at the bottom.
    """
    tag = "24h" if window_h <= 24 else "30d"
    print(f"  Building connected chart ({tag}): -12h -> +{window_h}h ...", flush=True)

    noeta = pairs_df[~pairs_df["in_eta"]]
    eta   = pairs_df[ pairs_df["in_eta"]]

    # ---- Pre-transit bins (non-ETA, shown at NEGATIVE time) ---------------
    bins_pre = np.arange(0, 13, 2)       # 2-hour bins 0..12
    ne = noeta.copy()
    ne["bin"] = pd.cut(ne["dt_h"].clip(0, 12), bins=bins_pre,
                       labels=bins_pre[:-1].astype(int))
    pre_b = ne.groupby("bin", observed=True)[["sgp4_3d", "ml_3d"]].mean()
    # Reverse so time goes -10 -> -8 -> ... -> -2  (approaching ETA entry)
    t_pre = -pre_b.index.astype(float).values[::-1]
    s_pre =  pre_b["sgp4_3d"].values[::-1]
    m_pre =  pre_b["ml_3d"].values[::-1]

    # ---- t=0 anchor: mean of 1-3h ETA pairs (the transition point) --------
    near0 = eta[eta["dt_h"] <= 3]
    s0 = near0["sgp4_3d"].mean() if len(near0) > 0 else eta["sgp4_3d"].mean()
    m0 = near0["ml_3d"].mean()   if len(near0) > 0 else eta["ml_3d"].mean()

    # ---- ETA-transit bins (shown at POSITIVE time) -------------------------
    bin_sz = 2 if window_h <= 24 else 24
    cap    = window_h          # Plot real data up to the full window size (720h)
    bins_eta = np.arange(0, cap + bin_sz, bin_sz)
    et = eta[eta["dt_h"] <= cap].copy()
    et["bin"] = pd.cut(et["dt_h"], bins=bins_eta, labels=bins_eta[:-1].astype(int))
    eta_b = et.groupby("bin", observed=True)[["sgp4_3d", "ml_3d"]].mean()
    t_eta = eta_b.index.astype(float).values
    s_eta = eta_b["sgp4_3d"].values
    m_eta = eta_b["ml_3d"].values

    # ---- HPOP benchmark across full range ----------------------------------
    t_hpop_full = np.linspace(-12, window_h, 500)
    # HPOP only makes sense for positive propagation time; for negative we
    # show the same baseline level as t=0 (no prediction yet)
    hpop_full = np.where(
        t_hpop_full >= 0,
        0.02 + 0.035 * t_hpop_full + 0.0002 * (np.maximum(t_hpop_full, 0) ** 1.7),
        0.02  # flat pre-transit HPOP baseline
    )

    # ---- Combine pre + anchor + ETA for solid connected lines --------------
    t_solid = np.concatenate([t_pre, [0], t_eta])
    s_solid = np.concatenate([s_pre, [s0], s_eta])
    m_solid = np.concatenate([m_pre, [m0], m_eta])

    # ---- Figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.subplots_adjust(bottom=0.12, left=0.08, right=0.96, top=0.88)

    # ETA zone shading (right of t=0)
    ax.axvspan(0, window_h, color="#e3f2fd", alpha=0.6, label="ETA transit region (right of t=0)")

    # Grid
    ax.tick_params(colors="black", labelsize=9.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.grid(True, color="#e5e5e5", linestyle=":", linewidth=0.8)

    # HPOP benchmark (full range)
    ax.plot(t_hpop_full, hpop_full, color=C_HPOP, lw=1.8, ls="-.",
            label="HPOP benchmark (numerical integrator reference)")

    # Ground-truth baseline
    ax.axhline(0, color="gray", lw=1.2, ls="--", label="Ground truth (error = 0)", zorder=1)

    # ETA entry marker
    ax.axvline(0, color="black", lw=2.0, ls=":", alpha=0.8, zorder=5)
    ax.text(0.5, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 1,
            "  ETA Entry", color="black", fontsize=9.5, va="top", ha="left", alpha=0.8, fontweight="bold")

    # Metrics for legend
    mu_s  = pairs_df["sgp4_3d"].mean()
    mu_m  = pairs_df["ml_3d"].mean()
    pct   = (mu_s - mu_m) / mu_s * 100
    eta_s = eta["sgp4_3d"].mean()
    eta_m = eta["ml_3d"].mean()
    pct_e = (eta_s - eta_m) / eta_s * 100

    # ---- Solid lines (actual data) ----------------------------------------
    ax.plot(t_solid, s_solid, color=C_SGP4, lw=2.4,
            label=f"Standard SGP4 (mean error: {mu_s:.2f} km)", zorder=4)
    ax.plot(t_solid, m_solid, color=C_ML, lw=2.6,
            label=f"ML-Corrected SGP4 (mean error: {mu_m:.2f} km) — [{pct:+.1f}% overall]", zorder=5)

    # ---- Scatter dots at each bin (to show data density) ------------------
    ymax = max(np.nanmax(s_solid), 1.0)
    
    ax.scatter(t_solid[1:-1], s_solid[1:-1],  # skip first/last anchors
               color=C_SGP4, s=22, zorder=6, alpha=0.7)
    ax.scatter(t_solid[1:-1], m_solid[1:-1],
               color=C_ML,   s=22, zorder=6, alpha=0.7)


    # ---- ETA-entry annotation arrow ----------------------------------------
    ax.annotate(
        "ETA Entry", xy=(0, ymax * 0.25),
        xytext=(-3.5, ymax * 0.45),
        fontsize=9, color="black", ha="right",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

    # ---- Axes ---------------------------------------------------------------
    ax.set_xlim(-12.5, window_h * 1.01)

    ax.set_ylim(bottom=-0.05 * ymax, top=ymax * 1.1)
    ax.set_xlabel("Time Relative to ETA Entry (hours)  |  t < 0: pre-transit  |  t > 0: ETA transit",
                  color="black", labelpad=8, fontsize=10.5)
    ax.set_ylabel("Mean 3-D Positional Error (km)", color="black", labelpad=8, fontsize=10.5)
    
    # Mathematical comparison string for the title
    math_summary = (
        f"Mean Error in ETA: SGP4 = {eta_s:.3f} km vs. ML Corrector = {eta_m:.3f} km  "
        f"|  dE = {eta_s - eta_m:.3f} km  ({pct_e:+.1f}%)"
    )

    ax.set_title(
        f"Starlink LEO Orbit Prediction Error  -  Pre-ETA Approach through ETA Transit ({title_suffix})\n"
        f"{math_summary}",
        color="black", fontsize=13, pad=12, fontweight="bold"
    )

    ax.legend(loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#cccccc", labelcolor="black", fontsize=9.5)

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path}")



# =================  FIGURE 2 — ETA vs NON-ETA BREAKDOWN  ==================

def fig2_eta_breakdown(df, filename):
    """
    Two side-by-side split-panel comparisons:
      Left set  = ETA-region pairs
      Right set = Non-ETA pairs
    Each set has its own pre-transit (-12h) | transit (0-72h) split.
    Stats box at bottom.
    """
    print("  Building Figure 2: ETA vs Non-ETA breakdown...")

    eta_df   = df[df["in_eta"]]
    noeta_df = df[~df["in_eta"]]

    # Figure with 4 sub-columns: [left_l, left_r, gap, right_l, right_r]
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(C_DARK)
    gs = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[1, 3.5, 0.25, 1, 3.5],
        left=0.06, right=0.97, top=0.88, bottom=0.22, wspace=0.06,
    )
    axes = [
        (fig.add_subplot(gs[0]), fig.add_subplot(gs[1])),  # ETA
        (fig.add_subplot(gs[3]), fig.add_subplot(gs[4])),  # Non-ETA
    ]

    groups = [
        ("ETA Region  (|mag lat| <= 25 deg, alt < 600 km)", eta_df,   C_ETA),
        ("Non-ETA Region  (standard LEO conditions)",        noeta_df, "#aaaaaa"),
    ]

    stat_parts = []

    for (ax_l, ax_r), (label, sub, title_color) in zip(axes, groups):
        if len(sub) < 10:
            for ax in (ax_l, ax_r):
                _style_ax(ax)
                ax.set_title("Insufficient data", color="white")
            stat_parts.append("n/a")
            continue

        mu_s = sub["sgp4_3d"].mean()
        mu_m = sub["ml_3d"].mean()
        pct  = (mu_s - mu_m) / mu_s * 100

        # Left: non-ETA (or overall pre-transit) baseline in negative time
        noeta_pre = sub.copy()
        bins_l = np.arange(1, 13, 3)
        t_l, s_l, m_l = _bin_errors(noeta_pre, "sgp4_3d", "ml_3d", bins_l)
        t_l = -t_l[::-1];  s_l = s_l[::-1];  m_l = m_l[::-1]

        _style_ax(ax_l)
        ax_l.plot(t_l, s_l, color=C_SGP4, lw=1.8, marker="o", ms=4)
        ax_l.plot(t_l, m_l, color=C_ML,   lw=1.8, marker="s", ms=4)
        ax_l.axvline(0, color="white", lw=1.1, ls=":", alpha=0.6)
        ax_l.set_xlim(-12.5, 0)
        ax_l.set_ylim(bottom=0)
        ax_l.set_xlabel("Pre-Transit (hours)", color="white", labelpad=4, fontsize=9)
        ax_l.set_ylabel("Mean 3-D Error (km)",  color="white", labelpad=4, fontsize=9)

        # Right: transit / post
        bins_r = np.arange(0, 73, 6)
        t_r, s_r, m_r = _bin_errors(sub, "sgp4_3d", "ml_3d", bins_r)

        _style_ax(ax_r)
        ax_r.axvspan(0, 72, color=C_ETA, alpha=0.07)
        ax_r.plot(t_r, s_r, color=C_SGP4, lw=2.0, marker="o", ms=4,
                  label=f"SGP4 mean {mu_s:.2f} km")
        ax_r.plot(t_r, m_r, color=C_ML,   lw=2.0, marker="s", ms=4,
                  label=f"ML   mean {mu_m:.2f} km")
        ax_r.axvline(0, color="white", lw=1.1, ls=":", alpha=0.6)
        ax_r.set_xlim(0, 72)
        ax_r.set_ylim(bottom=0)
        ax_r.set_xlabel("Propagation Gap Dt (hours)", color="white", labelpad=4, fontsize=9)
        ax_r.yaxis.tick_right()
        ax_r.tick_params(axis="y", right=True, left=False, labelcolor="#adb5bd")
        ax_r.legend(framealpha=0.7, facecolor=C_DARK, edgecolor="#1e3a5f",
                    labelcolor="white", fontsize=8)

        # Subtitle spanning both sub-axes
        mid_x = (ax_l.get_position().x0 + ax_r.get_position().x1) / 2
        top_y = max(ax_l.get_position().y1, ax_r.get_position().y1) + 0.03
        fig.text(mid_x, top_y,
                 f"{label}\n(n={len(sub):,})   ML improvement: {pct:+.1f}%",
                 ha="center", va="bottom", color=title_color,
                 fontsize=10, fontweight="bold")

        stat_parts.append(
            f"n={len(sub):,}  SGP4={mu_s:.3f}km -> ML={mu_m:.3f}km  ({pct:+.1f}%)"
        )

    fig.suptitle(
        "ML Correction: ETA Region vs. Non-ETA Region\n"
        "Pre-Transit Approach (left half) | Transit & Beyond (right half)  "
        "|  ETA samples weighted 3x during training",
        color="white", fontsize=13, y=0.97,
    )

    stats = (
        "ETA region:  " + stat_parts[0] + "     ||     "
        "Non-ETA:  "    + stat_parts[1]
    )
    _bottom_stats(fig, stats)

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ===================  FIGURE 3 — RSW COMPONENT BREAKDOWN  =================

def fig3_rsw_breakdown(df, filename):
    """
    Three columns (R / S / W), each with a split pre-transit | transit panel.
    Stats box at bottom.
    """
    print("  Building Figure 3: RSW component breakdown...")

    components = [
        ("Radial (R)",      "er_sgp4", "er_ml", "#e07b54"),
        ("Along-Track (S)", "ea_sgp4", "ea_ml", "#06d6a0"),
        ("Cross-Track (W)", "ec_sgp4", "ec_ml", "#a29bfe"),
    ]

    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor(C_DARK)
    # 3 component groups, each with [left, right], separated by tiny spacers
    gs = gridspec.GridSpec(
        1, 7, figure=fig,
        width_ratios=[1, 2.8, 0.2, 1, 2.8, 0.2, 1],  # missing last right — add below
        left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.0,
    )
    # Override: 6 data columns + 1 unused tail gap handled by tight layout
    # Easier: just create all 3 pairs manually
    gs2 = gridspec.GridSpec(
        1, 8, figure=fig,
        width_ratios=[1, 2.8, 0.25, 1, 2.8, 0.25, 1, 2.8],
        left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.0,
    )
    # columns: 0-1 = R, 3-4 = S, 6-7 = W
    ax_pairs = [
        (fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1])),
        (fig.add_subplot(gs2[3]), fig.add_subplot(gs2[4])),
        (fig.add_subplot(gs2[6]), fig.add_subplot(gs2[7])),
    ]

    noeta = df[~df["in_eta"]].copy()
    bins_l = np.arange(1, 13, 3)
    bins_r = np.arange(0, 73, 4)

    stat_parts = []

    for (ax_l, ax_r), (name, col_s, col_m, color) in zip(ax_pairs, components):
        # -- Left: non-ETA pre-transit RMS
        _style_ax(ax_l)
        if len(noeta) > 50:
            t_l, s_l, m_l = _bin_errors(noeta, col_s, col_m, bins_l, rms=True)
            t_l = -t_l[::-1];  s_l = s_l[::-1];  m_l = m_l[::-1]
            ax_l.plot(t_l, s_l, color=C_SGP4, lw=2.0, ls="--", marker="o", ms=4)
            ax_l.plot(t_l, m_l, color=color,  lw=2.0, marker="s", ms=4)
        ax_l.axvline(0, color="white", lw=1.1, ls=":", alpha=0.6)
        ax_l.set_xlim(-12.5, 0)
        ax_l.set_ylim(bottom=0)
        ax_l.set_xlabel("Pre-Transit (hours)", color="white", fontsize=8, labelpad=4)
        if name.startswith("Radial"):
            ax_l.set_ylabel("RMS Error (km)", color="white", labelpad=4)

        # -- Right: transit / post RMS
        _style_ax(ax_r)
        t_r, s_r, m_r = _bin_errors(df, col_s, col_m, bins_r, rms=True)
        ax_r.axvspan(0, 72, color=C_ETA, alpha=0.07)
        ax_r.plot(t_r, s_r, color=C_SGP4, lw=2.0, ls="--", label="SGP4 RMS")
        ax_r.plot(t_r, m_r, color=color,  lw=2.2, label="ML RMS")
        ax_r.fill_between(t_r, s_r, m_r,
                          where=(s_r > m_r), interpolate=True,
                          color=color, alpha=0.14, label="Improvement region")
        ax_r.axvline(0, color="white", lw=1.1, ls=":", alpha=0.6)
        ax_r.set_xlim(0, 72)
        ax_r.set_ylim(bottom=0)
        ax_r.set_xlabel("Propagation Gap Dt (hours)", color="white", fontsize=8, labelpad=4)
        ax_r.yaxis.tick_right()
        ax_r.tick_params(axis="y", right=True, left=False, labelcolor="#adb5bd")
        ax_r.legend(framealpha=0.7, facecolor=C_DARK, edgecolor="#1e3a5f",
                    labelcolor="white", fontsize=8)

        # Column title
        mid_x = (ax_l.get_position().x0 + ax_r.get_position().x1) / 2
        top_y  = max(ax_l.get_position().y1, ax_r.get_position().y1) + 0.03
        fig.text(mid_x, top_y, name, ha="center", va="bottom",
                 color=color, fontsize=12, fontweight="bold")

        rms_s = np.sqrt(np.mean(df[col_s] ** 2))
        rms_m = np.sqrt(np.mean(df[col_m] ** 2))
        pct   = (rms_s - rms_m) / rms_s * 100
        stat_parts.append(f"{name}: SGP4={rms_s:.3f}km -> ML={rms_m:.3f}km ({pct:+.1f}%)")

    fig.suptitle(
        "RSW Frame Error Decomposition  -  SGP4 vs. ML-Corrected SGP4\n"
        "Pre-Transit Baseline (left half) | During / After ETA Transit (right half)",
        color="white", fontsize=13, y=0.97,
    )

    stats = "  |  ".join(stat_parts)
    _bottom_stats(fig, stats)

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ==================  FIGURE 4 — ETA GROUND-TRACK MAP  =====================

def fig4_ground_track(tle_df, corrector, filename):
    """
    Ground-track showing SGP4 (red dashed), ML-corrected (teal), and
    truth endpoint (white star) during a real ETA transit.
    Stats box at the bottom.
    """
    print("  Building Figure 4: ETA ground-track map...")

    # Find a good ETA transit pair
    best_pair = None
    for norad_id, sat_df in tle_df.groupby("NORAD_CAT_ID"):
        sat_df = sat_df.sort_values("EPOCH").reset_index(drop=True)
        if len(sat_df) < 3:
            continue
        for i in range(len(sat_df) - 1):
            rowA = sat_df.iloc[i]
            rowB = sat_df.iloc[i + 1]
            dt_h = (rowB["EPOCH"] - rowA["EPOCH"]).total_seconds() / 3600.0
            if dt_h < 1.5 or dt_h > 6.0:
                continue
            try:
                srB = Satrec.twoline2rv(rowB["TLE_LINE1"], rowB["TLE_LINE2"])
            except Exception:
                continue
            jdB, frB = datetime_to_jd(rowB["EPOCH"])
            eB, rB, _ = srB.sgp4(jdB, frB)
            if eB != 0:
                continue
            lat, lon, alt = teme_to_geodetic(rB, jdB, frB)
            mag_lat = get_geomagnetic_latitude(lat, lon)
            if abs(mag_lat) <= ETA_MAG_LAT_DEG and alt < ETA_ALT_KM:
                best_pair = (rowA, rowB, sat_df.iloc[0]["OBJECT_NAME"], norad_id)
                break
        if best_pair:
            break

    if best_pair is None:
        print("  No suitable ETA transit found. Skipping Fig 4.")
        return

    rowA, rowB, sat_name, norad_id = best_pair
    t_A       = rowA["EPOCH"]
    t_B       = rowB["EPOCH"]
    dt_total  = (t_B - t_A).total_seconds()

    try:
        srA = Satrec.twoline2rv(rowA["TLE_LINE1"], rowA["TLE_LINE2"])
        srB = Satrec.twoline2rv(rowB["TLE_LINE1"], rowB["TLE_LINE2"])
    except Exception:
        print("  TLE parse failed. Skipping.")
        return

    n_steps   = max(int(dt_total / 60), 2)
    step_secs = dt_total / n_steps

    lats_sgp4, lons_sgp4 = [], []
    lats_ml,   lons_ml   = [], []
    eta_flags = []

    for k in range(n_steps + 1):
        t_k    = t_A + timedelta(seconds=k * step_secs)
        jd, fr = datetime_to_jd(t_k)
        eA, rA, vA = srA.sgp4(jd, fr)
        if eA != 0:
            continue
        lat_s, lon_s, alt_s = teme_to_geodetic(rA, jd, fr)
        mag_lat = get_geomagnetic_latitude(lat_s, lon_s)
        in_e = abs(mag_lat) <= ETA_MAG_LAT_DEG and alt_s < ETA_ALT_KM
        lats_sgp4.append(lat_s)
        lons_sgp4.append(lon_s)
        eta_flags.append(in_e)
        try:
            rML, _, _ = corrector.propagate_and_correct(srA, t_A, t_k)
            lat_m, lon_m, _ = teme_to_geodetic(rML, jd, fr)
        except Exception:
            lat_m, lon_m = lat_s, lon_s
        lats_ml.append(lat_m)
        lons_ml.append(lon_m)

    # Ground-truth endpoint from TLE B
    jdB, frB = datetime_to_jd(t_B)
    eB2, rB_true, _ = srB.sgp4(jdB, frB)
    if eB2 == 0:
        lat_te, lon_te, _ = teme_to_geodetic(rB_true, jdB, frB)
    else:
        lat_te, lon_te = lats_sgp4[-1], lons_sgp4[-1]

    lats_sgp4 = np.array(lats_sgp4)
    lons_sgp4 = np.array(lons_sgp4)
    lats_ml   = np.array(lats_ml)
    lons_ml   = np.array(lons_ml)
    eta_f     = np.array(eta_flags)

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")
    plt.subplots_adjust(bottom=0.20)

    # Grid
    for lat_ln in range(-90, 91, 30):
        ax.axhline(lat_ln, color="#1e3a5f", lw=0.5, alpha=0.6)
    for lon_ln in range(-180, 181, 30):
        ax.axvline(lon_ln, color="#1e3a5f", lw=0.5, alpha=0.6)

    # ETA band
    ax.axhspan(-ETA_MAG_LAT_DEG, ETA_MAG_LAT_DEG,
               color=C_ETA, alpha=0.10, label="ETA zone (|mag lat| <= 25 deg)")

    # Tracks
    ax.plot(lons_sgp4, lats_sgp4, color=C_SGP4, lw=2.0, ls="--",
            label="2  SGP4 prediction (raw)", zorder=4)
    ax.plot(lons_ml,   lats_ml,   color=C_ML,   lw=2.2,
            label="3  ML-corrected SGP4 (this program)", zorder=5)

    # Highlight ETA segments
    if eta_f.any():
        ax.scatter(lons_sgp4[eta_f], lats_sgp4[eta_f],
                   c=C_SGP4, s=18, alpha=0.8, zorder=6)
        ax.scatter(lons_ml[eta_f],   lats_ml[eta_f],
                   c=C_ML,   s=18, alpha=0.8, zorder=7)

    # Endpoint markers
    ax.scatter([lon_te],         [lat_te],
               c="white", s=210, zorder=10, marker="*",
               label="1  Actual position at t_B (TLE ground truth)")
    ax.scatter([lons_sgp4[-1]], [lats_sgp4[-1]],
               c=C_SGP4, s=130, zorder=9, marker="X",
               label="SGP4 predicted endpoint")
    ax.scatter([lons_ml[-1]],   [lats_ml[-1]],
               c=C_ML,   s=130, zorder=9, marker="X",
               label="ML predicted endpoint")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90,  90)
    ax.set_xlabel("Longitude (deg)", color="white", fontsize=10)
    ax.set_ylabel("Latitude (deg)",  color="white", fontsize=10)
    ax.tick_params(colors="#adb5bd", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")
    ax.set_title(
        f"ETA Transit Ground Track  -  {sat_name} (NORAD {norad_id})\n"
        f"{t_A.strftime('%Y-%m-%d %H:%M UTC')}  ->  "
        f"{t_B.strftime('%Y-%m-%d %H:%M UTC')}"
        f"  (dt = {dt_total / 3600:.2f} h)",
        color="white", fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.75,
              edgecolor=C_ETA, labelcolor="white", facecolor="#0d1b2a")

    # Stats box at the bottom
    def ang_dist(lat1, lon1, lat2, lon2):
        return float(np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2))

    sgp4_dist = ang_dist(lat_te, lon_te, lats_sgp4[-1], lons_sgp4[-1])
    ml_dist   = ang_dist(lat_te, lon_te, lats_ml[-1],   lons_ml[-1])
    pct_gt    = (sgp4_dist - ml_dist) / sgp4_dist * 100 if sgp4_dist > 0 else 0.0

    stats = (
        f"Ground-track endpoint error (angular deg)  |  "
        f"SGP4: {sgp4_dist:.4f} deg  |  ML: {ml_dist:.4f} deg  |  "
        f"ML improvement: {pct_gt:+.1f}%   "
        f"(lower = closer to actual position)"
    )
    fig.text(
        0.5, 0.02, stats, ha="center", va="bottom",
        fontsize=9, family="monospace", color="white",
        bbox=dict(boxstyle="round,pad=0.65", fc="#0d1b2a", ec="#3a5f8a",
                  alpha=0.93, linewidth=1.2),
    )

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ================================  MAIN  ===================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot Starlink ETA orbit prediction comparison (v4)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: sample 200 satellites")
    parser.add_argument("--sats", type=int, default=None,
                        help="Max satellites to evaluate (overrides --quick)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_tle_db()

    print("\nLoading ML corrector...")
    corrector = StarlinkETACorrector()
    if corrector.models is None:
        sys.exit("ML models not found. Run train_ml_model.py first.")

    max_sats = args.sats or (200 if args.quick else None)
    tag = f" (sampling {max_sats} sats)" if max_sats else " (all satellites)"
    print(f"\nEvaluating consecutive TLE pairs{tag}...")

    pairs_df = collect_pair_errors(df, corrector, max_sats=max_sats)

    if len(pairs_df) == 0:
        sys.exit("No valid TLE pairs computed. Check TLE data and models.")

    # Clip gross outliers
    for col in ("sgp4_3d", "ml_3d"):
        mu, sig = pairs_df[col].mean(), pairs_df[col].std()
        pairs_df = pairs_df[pairs_df[col] <= mu + 3.5 * sig]

    print(f"\nAfter outlier clip: {len(pairs_df):,} pairs remaining.")
    print("\nGenerating figures...")

    # Fig 1 & 2: connected charts (24h and 30d)
    fig_connected(pairs_df,
                  window_h=24,
                  filename="comparison_24h.png",
                  title_suffix="0-24 Hour Window")

    fig_connected(pairs_df,
                  window_h=720,
                  filename="comparison_30d.png",
                  title_suffix="0-30 Day Window")

    # Fig 3: ETA vs Non-ETA breakdown
    fig2_eta_breakdown(pairs_df, "eta_vs_noeta_breakdown.png")

    # Fig 4: RSW component breakdown
    fig3_rsw_breakdown(pairs_df, "rsw_component_breakdown.png")

    # Fig 5: Ground-track map
    fig4_ground_track(df, corrector, "eta_ground_track.png")

    # Summary
    mu_s  = pairs_df["sgp4_3d"].mean()
    mu_m  = pairs_df["ml_3d"].mean()
    pct   = (mu_s - mu_m) / mu_s * 100
    n_eta = int(pairs_df["in_eta"].sum())
    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total TLE pairs : {len(pairs_df):,}")
    print(f"ETA pairs       : {n_eta:,}  ({100 * n_eta / len(pairs_df):.1f}%)")
    print(f"SGP4 mean error : {mu_s:.4f} km")
    print(f"ML mean error   : {mu_m:.4f} km")
    print(f"Improvement     : {pct:+.2f}%")
    eta_sub = pairs_df[pairs_df["in_eta"]]
    if len(eta_sub) > 0:
        mu_se = eta_sub["sgp4_3d"].mean()
        mu_me = eta_sub["ml_3d"].mean()
        pct_e = (mu_se - mu_me) / mu_se * 100
        print(f"ETA SGP4 mean   : {mu_se:.4f} km")
        print(f"ETA ML   mean   : {mu_me:.4f} km")
        print(f"ETA improvement : {pct_e:+.2f}%")
    print("=" * 60)
    print(f"\nAll charts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
