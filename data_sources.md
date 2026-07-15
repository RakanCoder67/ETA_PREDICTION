# ETA Prediction Project: Data Sources

This file documents the exact endpoints, archives, and APIs used to gather both the orbital elements and space weather indicators for training and running the ETA Machine Learning Correction Model.

---

## 1. Orbital Elements (Satellites)

### Space-Track TLE History
* **Provider**: U.S. Space Command (via Space-Track.org)
* **Website**: [https://www.space-track.org](https://www.space-track.org)
* **Endpoints Used**:
  * `https://www.space-track.org/basicspacedata/query/class/gp_history` (Historical Two-Line Element sets)
  * `https://www.space-track.org/basicspacedata/query/class/gp` (Live General Perturbations TLE sets)
* **Query Parameters**: Filtered for `OBJECT_NAME/~~STARLINK` and epochs covering the active training window.
* **Format**: Two-Line Element (TLE) format inside CSV rows.

---

## 2. Space Weather & Solar Indicators

### Live NOAA Space Weather Prediction Center (SWPC)
* **Provider**: NOAA / National Weather Service
* **Website**: [https://www.swpc.noaa.gov](https://www.swpc.noaa.gov)
* **Endpoints Used**:
  * **Planetary Kp Index (1-minute)**: `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json`
  * **GOES Solar X-Ray Flux (7-day)**: `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json`
  * **Solar Radio Flux F10.7 (30-day)**: `https://services.swpc.noaa.gov/products/10cm-flux-30-day.json`
* **Format**: JSON arrays containing timestamps (`time_tag`) and numerical indices.

---

## 3. Historical Space Weather Archives

### GFZ German Research Centre for Geosciences
* **Provider**: Helmholtz Centre Potsdam (GFZ)
* **Website**: [https://www.gfz-potsdam.de](https://www.gfz-potsdam.de)
* **Dataset**: Historical Kp, Ap, and F10.7 indices (spanning 1932 to present)
* **Local Source File**: `Kp_ap_Ap_SN_F107_since_1932.txt`
* **Use Case**: Used to backfill space weather features matching historical satellite epochs outside the NOAA 7-day live window.

### SILSO Sunspot Number Database
* **Provider**: World Data Center for the Production, Preservation and Dissemination of the International Sunspot Number (Royal Observatory of Belgium)
* **Website**: [https://www.sidc.be/silso/datafiles](https://www.sidc.be/silso/datafiles)
* **Dataset**: Daily Sunspot Number (total version 2.0)
* **Local Source File**: `SN_d_tot_V2.0.txt`
* **Use Case**: Integrated as a proxy for long-term solar cycle intensity changes.

### Penticton Radio Solar Flux Archive
* **Provider**: National Research Council (NRC) Canada / Dominion Radio Astrophysical Observatory (DRAO)
* **Dataset**: Penticton F10.7 observation records (daily mean)
* **Local Source File**: `fluxtable.txt`
* **Use Case**: Secondary source for highly calibrated 10.7cm solar radio flux records.
