import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LivePlotCallback(xgb.callback.TrainingCallback):
    def __init__(self, target, models_dict, base_dir, features):
        self.target = target
        self.models_dict = models_dict
        self.base_dir = base_dir
        self.features = features

    def after_iteration(self, model, epoch, evals_log):
        if epoch > 0 and epoch % 50 == 0:
            self.models_dict[self.target] = model
            model_path = os.path.join(self.base_dir, 'models', 'sgp4_correction_models.pkl')
            features_path = os.path.join(self.base_dir, 'models', 'model_features.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.models_dict, model_path)
            joblib.dump(self.features, features_path)
            
            # Run propagation script in background to update charts
            import subprocess
            subprocess.Popen([sys.executable, os.path.join(self.base_dir, 'propagate_30d.py')],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f" -> Epoch {epoch}: Live plot updated!")
        return False

def main():
    base_dir = r"c:\Users\Rakan Alghamdi\OneDrive - Lausanne Collegiate School\Documents\ETA_PREDICTION"
    data_path = os.path.join(base_dir, 'ml_training_dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run generate_training_data.py first.")
        return
        
    print("Loading training data...")
    df = pd.read_csv(data_path)
    
    features = [
        'dt_hours', 'BSTAR', 'INCLINATION', 'ECCENTRICITY', 
        'altitude', 'latitude', 'longitude', 'magnetic_lat', 'local_time',
        'magnetic_lt', 'eta_intensity', 'in_eta',
        'Ap', 'F107', 'Kp_index', 'Dst', 'solar_cycle_phase', 'Xray_short', 'Xray_long'
    ]
                
    targets = ['err_radial', 'err_along', 'err_cross']
    
    df = df.dropna(subset=features + targets)
    if len(df) == 0:
        print("No valid training samples found.")
        return
        
    X = df[features]
    y = df[targets]
    
    # Horizon-weighted training: give samples within the first day (dt_hours <= 24) a 20x higher weight
    # and give ETA samples a 3x higher weight, stacking them for extreme short-term transit accuracy.
    sample_weights = np.ones(len(df))
    sample_weights = np.where(df['in_eta'] == 1, sample_weights * 3.0, sample_weights)
    sample_weights = np.where(df['dt_hours'] <= 24.0, sample_weights * 20.0, sample_weights)
    
    X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
        X, y, sample_weights, np.arange(len(df)), test_size=0.2, random_state=42
    )
    
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, w_train, test_size=0.1, random_state=42
    )
    
    print(f"Training on {len(X_tr)} samples, validating on {len(X_val)} samples, testing on {len(X_test)} samples.")
    
    models = {}
    
    # Load existing models if present to keep them populated for background propagation
    model_path = os.path.join(base_dir, 'models', 'sgp4_correction_models.pkl')
    if os.path.exists(model_path):
        try:
            models = joblib.load(model_path)
        except:
            pass

    for target in targets:
        print(f"\nTraining model for {target}...")
        model = xgb.XGBRegressor(
            n_estimators=1200, 
            learning_rate=0.015, 
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit model
        model.fit(
            X_tr, y_tr[target], 
            sample_weight=w_tr,
            eval_set=[(X_val, y_val[target])],
            sample_weight_eval_set=[w_val],
            verbose=100
        )
        # Remove callback reference to prevent pickling errors on unpickle
        model.set_params(callbacks=None)
        models[target] = model
        
        y_pred = model.predict(X_test)
        sgp4_rmse = np.sqrt(np.mean(y_test[target]**2))
        hybrid_rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
        improvement_pct = (sgp4_rmse - hybrid_rmse) / sgp4_rmse * 100
        
        print(f"Overall SGP4 Baseline RMSE ({target}): {sgp4_rmse:.4f} km")
        print(f"Overall Hybrid ML RMSE     ({target}): {hybrid_rmse:.4f} km")
        print(f"Overall Improvement: {improvement_pct:.1f}%")
        
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    joblib.dump(models, model_path)
    joblib.dump(features, os.path.join(base_dir, 'models', 'model_features.pkl'))
    print(f"\nModels successfully saved to {model_path}")

if __name__ == "__main__":
    main()
