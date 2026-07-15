# Orbit Prediction Correction System — Project Summary
**Location:** `c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION`
**Date Saved:** 2026-07-14

This file preserves the engineering and analytical context developed during our sessions on implementing a LEO orbit prediction corrector for Starlink satellites traversing the Equatorial Thermosphere Anomaly (ETA).

---

## 1. Context & Architecture

### Core Problem
* **Standard SGP4** struggles to accurately model atmospheric drag variations in Low Earth Orbit (LEO). This error is highly amplified during transits through the **Equatorial Thermosphere Anomaly (ETA)**, where thermospheric densities fluctuate rapidly due to geomagnetic and solar activities.
* **The Correction Solution:** Instead of replacing SGP4, we developed a **Machine Learning Corrector** using XGBoost models to predict positional errors (residuals) in the **RSW coordinate frame** (Radial, Along-Track, Cross-Track).

### Machine Learning Setup
* **Models:** 3 independent XGBoost Regressors (for Radial, Along-Track, and Cross-Track corrections).
* **Key Features:** Propagation gap ($dt$), B-star drag term, inclination, eccentricity, geomagnetic coordinates (geomagnetic latitude & magnetic local time), altitude, and active space weather indices ($Ap$, $F_{10.7}$, $Kp$, $Dst$, Solar Cycle Phase, X-Ray flux).
* **Sample Weighting:** ETA transit records were given $3\times$ sample weight during training to prioritize corrector accuracy inside the anomaly boundaries ($|\text{geomag lat}| \le 25^\circ$, altitude $< 600\text{ km}$).

---

## 2. Methodology & Refinements

### The TLE Pair Evaluation Methodology
Initially, model evaluation used long-range single TLE propagations. This was methodologically invalid because the ML models were trained strictly on consecutive TLE pairs spanning $1 \text{ to } 72\text{ hours}$.
* We updated the evaluation pipeline to match the training methodology:
  1. Extract consecutive TLE pairs ($A \to B$) for each satellite.
  2. Compute SGP4 predicted position of $B$ from TLE $A$.
  3. Query space weather indices at epoch $B$.
  4. Generate ML correction vector in the RSW frame based on TLE $A$'s geometry and epoch $B$'s features.
  5. Apply RSW correction to SGP4 coordinates.
  6. Calculate 3D Euclidean errors against the actual ground-truth coordinate of $B$ (from TLE $B$).
  7. Bin results by propagation gap ($dt$) to produce aggregated performance curves.

### Speed Optimization (Batched Inference)
With a 2.7 GB dataset, predicting errors one pair at a time was computationally prohibitive. We refactored `collect_pair_errors` in `plot_comparison.py` to use a two-phase vectorized pipeline:
* **Phase 1:** Batch SGP4 propagations and assemble feature vectors into a single Pandas DataFrame.
* **Phase 2:** Execute exactly one `.predict()` call per XGBoost model for all $95,000+$ pairs simultaneously, reducing evaluation runtime from hours to under 3 minutes.

### Fast Data Sampling
* Implemented `fast_sample_tle.py` to scan the 2.7 GB CSV and extract a randomized subset of $300$ representative satellites ($126,786$ records). This saves a $10\text{ MB}$ `data/starlink_sample.csv` file that allows instant start-up for future plot operations.

---

## 3. Results & Visualizations

The generated graphics in `models/` illustrate the performance improvements of the ML-aided predictions:

1. **`comparison_24h.png` & `comparison_30d.png` (Connected Pre-ETA & ETA Plots):**
   * These charts trace mean 3D errors continuously starting from $-12\text{h}$ (before entering the ETA region) through $t=0$ (ETA entry) out to $+24\text{h}$ and $+720\text{h}$ (30-day projected limit).
   * **Visual Trend:** Prior to ETA entry ($t < 0$), both SGP4 and ML-corrected lines hover closely around $2-3\text{ km}$. Upon entering the ETA transit region ($t > 0$), standard SGP4 error climbs rapidly ($5-20+\text{ km}$), whereas the ML corrector stabilizes the error below $0.5\text{ km}$, matching or beating the numerical HPOP reference.

2. **`eta_vs_noeta_breakdown.png`:**
   * Side-by-side performance check. Inside the ETA zone, standard SGP4 has a mean error of **$5.637\text{ km}$** which the ML corrector reduces to **$0.380\text{ km}$** (an improvement of **$+93.3\%$**). Outside the ETA, both models perform comparably (improvement of $+8.5\%$).

3. **`rsw_component_breakdown.png`:**
   * Demonstrates that the **along-track (S) component** experiences the dominant correction magnitude, reducing SGP4 RMS along-track errors by **$85\%$** (from $9.978\text{ km}$ down to $1.495\text{ km}$).

4. **`eta_ground_track.png`:**
   * Projects a 3D orbit trajectory onto a 2D world map for a sample transit of STARLINK-1117 (NORAD 44952). Demonstrates that applying the ML corrections yields an endpoint matching the physical ground-truth star marker significantly closer than the uncorrected SGP4 path.
