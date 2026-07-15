import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Auto-install matplotlib if not present
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    import subprocess
    print("matplotlib not found. Installing it now for visual results...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        print("matplotlib successfully installed.")
    except Exception as e:
        print(f"Failed to install matplotlib automatically: {e}")
        print("Please install it manually using: pip install matplotlib")
        sys.exit(1)

from sgp4.api import Satrec
from starlink_eta_corrector import StarlinkETACorrector, datetime_to_jd, get_rsw_basis

def generate_comparison_plots(norad_id=44714):
    base_dir = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
    tle_path = os.path.join(base_dir, 'data', 'starlink_history.csv')
    
    if not os.path.exists(tle_path):
        print("Error: TLE history not found. Please run fetch_live_data.py first.")
        return
        
    df = pd.read_csv(tle_path)
    sat_df = df[df['NORAD_CAT_ID'] == norad_id].sort_values('EPOCH')
    
    if len(sat_df) < 2:
        print(f"Error: Satellite NORAD ID {norad_id} must have at least 2 TLEs in database to compare paths.")
        print(f"Available NORAD IDs: {list(df['NORAD_CAT_ID'].unique()[:10])}")
        return
        
    # We take TLE A and TLE B (consecutive updates)
    row_A = sat_df.iloc[-2]
    row_B = sat_df.iloc[-1]
    
    t_A = pd.to_datetime(row_A['EPOCH'], utc=True)
    t_B = pd.to_datetime(row_B['EPOCH'], utc=True)
    
    dt_hours = (t_B - t_A).total_seconds() / 3600.0
    print(f"\nAnalyzing Starlink {row_A['OBJECT_NAME']} (NORAD {norad_id})")
    print(f"Start Epoch (TLE A): {t_A}")
    print(f"End Epoch (TLE B):   {t_B}")
    print(f"Total time delta:    {dt_hours:.2f} hours")
    
    # Initialize corrector
    corrector = StarlinkETACorrector()
    
    satrec_A = Satrec.twoline2rv(row_A['TLE_LINE1'], row_A['TLE_LINE2'])
    satrec_B = Satrec.twoline2rv(row_B['TLE_LINE1'], row_B['TLE_LINE2'])
    
    # We will propagate from t_A to t_B in 100 steps
    steps = 100
    times = [t_A + timedelta(seconds=i * (t_B - t_A).total_seconds() / steps) for i in range(steps + 1)]
    
    r_actuals = []
    r_sgp4s = []
    r_hybrids = []
    
    time_offsets = []
    sgp4_errors_along = []
    hybrid_errors_along = []
    
    sgp4_errors_3d = []
    hybrid_errors_3d = []
    
    in_etas = []
    
    for t in times:
        jd, fr = datetime_to_jd(t)
        
        # 1. "Actual" path (piecewise propagation of the closest true TLE for highest accuracy)
        # SGP4 is highly accurate near its epoch, so using TLE A for first half and TLE B for second half
        # represents the most realistic 'truth' trajectory without a high-fidelity numerical ephemeris.
        half_time = t_A + (t_B - t_A) / 2
        if t < half_time:
            _, r_act, _ = satrec_A.sgp4(jd, fr)
        else:
            _, r_act, _ = satrec_B.sgp4(jd, fr)
            
        # 2. Raw SGP4 path (Other programs' prediction: propagated from TLE A all the way)
        _, r_sgp4, v_sgp4 = satrec_A.sgp4(jd, fr)
        
        # 3. Hybrid ML corrected path (This program's prediction)
        r_hybrid, _, info = corrector.propagate_and_correct(satrec_A, t_A, t)
        
        r_actuals.append(r_act)
        r_sgp4s.append(r_sgp4)
        r_hybrids.append(r_hybrid)
        
        # Convert to numpy arrays
        r_act = np.array(r_act)
        r_sgp4 = np.array(r_sgp4)
        r_hybrid = np.array(r_hybrid)
        v_sgp4 = np.array(v_sgp4)
        
        # Calculate error metrics
        u_r, u_s, u_w = get_rsw_basis(r_sgp4, v_sgp4)
        
        dr_sgp4 = r_act - r_sgp4
        dr_hybrid = r_act - r_hybrid
        
        along_err_sgp4 = abs(np.dot(dr_sgp4, u_s))
        along_err_hybrid = abs(np.dot(dr_hybrid, u_s))
        
        sgp4_errors_along.append(along_err_sgp4)
        hybrid_errors_along.append(along_err_hybrid)
        
        sgp4_errors_3d.append(np.linalg.norm(dr_sgp4))
        hybrid_errors_3d.append(np.linalg.norm(dr_hybrid))
        
        dt_elapsed = (t - t_A).total_seconds() / 3600.0
        time_offsets.append(dt_elapsed)
        in_etas.append(info['in_eta'])
        
    r_actuals = np.array(r_actuals)
    r_sgp4s = np.array(r_sgp4s)
    r_hybrids = np.array(r_hybrids)
    
    # Define plot outputs
    fig_error_path = os.path.join(base_dir, 'models', 'prediction_error_comparison.png')
    fig_3d_path = os.path.join(base_dir, 'models', 'orbital_trajectory_comparison.png')
    
    print("\nGenerating Plots...")
    
    # Plot 1: Positional Error over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_offsets, sgp4_errors_along, label="Standard SGP4 Prediction Error (Other Programs)", color='#ff5555', linewidth=2)
    plt.plot(time_offsets, hybrid_errors_along, label="ML-Corrected SGP4 Prediction Error (THIS Program)", color='#22cc88', linewidth=2.5)
    plt.axhline(0, color='black', linestyle='--', label="Actual Trajectory (Baseline Truth)", alpha=0.7)
    
    # Highlight ETA region transits
    eta_intervals = []
    in_eta_run = False
    start_eta = 0.0
    for idx, in_eta in enumerate(in_etas):
        if in_eta and not in_eta_run:
            start_eta = time_offsets[idx]
            in_eta_run = True
        elif not in_eta and in_eta_run:
            plt.axvspan(start_eta, time_offsets[idx-1], color='#3399ff', alpha=0.15, label="ETA Region Transit" if "ETA Region Transit" not in plt.gca().get_legend_handles_labels()[1] else "")
            in_eta_run = False
    if in_eta_run:
        plt.axvspan(start_eta, time_offsets[-1], color='#3399ff', alpha=0.15, label="ETA Region Transit" if "ETA Region Transit" not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Orbital Along-Track Position Error Over Time\nStarlink {row_A['OBJECT_NAME']} (NORAD {norad_id})", fontsize=14, pad=15)
    plt.xlabel("Propagation Time elapsed (Hours)", fontsize=12)
    plt.ylabel("Positional Error Deviation (Kilometers)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(fig_error_path, dpi=150)
    print(f"- Saved Error Comparison Plot to: {fig_error_path}")
    
    # Plot 2: 3D Orbital Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # We plot a slice of the trajectory (e.g. first 20 steps to show differences clearly)
    plot_slice = slice(0, steps + 1)
    
    ax.plot(r_actuals[plot_slice, 0], r_actuals[plot_slice, 1], r_actuals[plot_slice, 2], 
            label="Actual Location (Greenwich/TLE reference)", color='black', linewidth=2.5, zorder=5)
    ax.plot(r_sgp4s[plot_slice, 0], r_sgp4s[plot_slice, 1], r_sgp4s[plot_slice, 2], 
            label="Other Programs' Prediction (Raw SGP4)", color='#ff5555', linestyle='--', linewidth=1.5, zorder=3)
    ax.plot(r_hybrids[plot_slice, 0], r_hybrids[plot_slice, 1], r_hybrids[plot_slice, 2], 
            label="This Program's Prediction (ML-Corrected)", color='#22cc88', linestyle='-.', linewidth=2.0, zorder=4)
            
    # Draw Earth sphere (for reference)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = 6378.137 * np.cos(u) * np.sin(v)
    y_earth = 6378.137 * np.sin(u) * np.sin(v)
    z_earth = 6378.137 * np.cos(v)
    # We only show the surface faintly
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='#99ccff', alpha=0.08)
    
    # Set labels
    ax.set_title(f"3D Orbital Trajectory Path Comparison\nStarlink {row_A['OBJECT_NAME']} (NORAD {norad_id})", fontsize=14, pad=15)
    ax.set_xlabel("TEME X Position (km)", fontsize=10)
    ax.set_ylabel("TEME Y Position (km)", fontsize=10)
    ax.set_zlabel("TEME Z Position (km)", fontsize=10)
    
    # Equal aspect ratio for 3D axis
    # Calculate limits
    all_x = np.concatenate([r_actuals[:, 0], r_sgp4s[:, 0], r_hybrids[:, 0]])
    all_y = np.concatenate([r_actuals[:, 1], r_sgp4s[:, 1], r_hybrids[:, 1]])
    all_z = np.concatenate([r_actuals[:, 2], r_sgp4s[:, 2], r_hybrids[:, 2]])
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_3d_path, dpi=150)
    print(f"- Saved 3D Trajectory Comparison Plot to: {fig_3d_path}")
    
    # Print numerical comparison
    final_sgp4_err = sgp4_errors_3d[-1]
    final_hybrid_err = hybrid_errors_3d[-1]
    improvement = (final_sgp4_err - final_hybrid_err) / final_sgp4_err * 100
    
    print("\n" + "="*55)
    print(f"TRAJECTORY POSITION ACCURACY REPORT ({dt_hours:.1f}-HOUR PROPAGATION)")
    print("="*55)
    print(f"Standard Propagators (SGP4) Position Error: {final_sgp4_err:.3f} km")
    print(f"This Program's Corrected Position Error:   {final_hybrid_err:.3f} km")
    print(f"Accuracy Improvement (Error Reduced by):     {improvement:.1f}%")
    print("="*55)

if __name__ == "__main__":
    parser = argparse_parser = argparse = None
    # Check if a custom NORAD ID was passed
    nid = 44714
    if len(sys.argv) > 1:
        try:
            nid = int(sys.argv[1])
        except ValueError:
            pass
    generate_comparison_plots(nid)
