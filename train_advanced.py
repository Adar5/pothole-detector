import pandas as pd
import numpy as np
import os
import joblib
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
MODEL_NAME = 'speed_breaker_model.pkl'

def apply_lowpass_filter(data, cutoff=10, fs=100, order=4):
    # This filters out any vibration faster than 10Hz (engine buzz)
    # leaving only the slow, physical movements of the bike (bumps)
    if len(data) <= 15: return data.values if isinstance(data, pd.Series) else data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    try:
        return filtfilt(b, a, data)
    except:
        return data.values if isinstance(data, pd.Series) else data

# --- 1. ADVANCED FEATURE EXTRACTION (Raw G-Force Edition) ---
def extract_features(df):
    # 1. Identify Axes
    axes = ['accel_x', 'accel_y', 'accel_z']
    if 'accel_x' not in df.columns:
        found_cols = [c for c in df.columns if c.lower().endswith(('x', 'y', 'z')) and 'gyro' not in c.lower()]
        if len(found_cols) >= 3: axes = found_cols[:3]
        else: return None
            
    means = df[axes].mean()
    vert_axis = means.abs().idxmax()
    horiz_axes = [c for c in axes if c != vert_axis]
    
    # 2. Extract Raw Signals
    raw_z = df[vert_axis] - means[vert_axis]
    raw_y = df[horiz_axes[0]] - means[horiz_axes[0]] # Using one of the horizontal axes
    
    # 3. APPLY BUTTERWORTH FILTER (Erase Engine Noise)
    clean_z = apply_lowpass_filter(raw_z)
    clean_y = apply_lowpass_filter(raw_y)
    
    # 4. Extract Features from the CLEANED 3D data
    max_val = clean_z.max()
    min_val = clean_z.min()
    
    n = len(clean_z)
    if n == 0: return None
    
    freq_values = fft(clean_z)
    fft_magnitude = np.abs(freq_values)[:n//2] 
    total_energy = np.sum(fft_magnitude)
    if total_energy == 0: total_energy = 1
    
    # Assuming standard 100Hz sampling rate for training data
    fs = 100
    idx_5hz = int(5 * (n / fs))
    idx_10hz = int(10 * (n / fs))
    
    features = {
        # -- Primary Z-Axis Features --
        'max_val': max_val,
        'min_val': min_val,
        'range': max_val - min_val,
        'std_dev': np.std(clean_z),
        'p95': np.percentile(np.abs(clean_z), 95),
        'zero_crossings': (np.diff(np.sign(clean_z)) != 0).sum(),
        'low_freq_ratio': np.sum(fft_magnitude[:idx_5hz]) / total_energy,   
        'high_freq_ratio': np.sum(fft_magnitude[idx_10hz:]) / total_energy, 
        'dominant_freq_mag': np.max(fft_magnitude) ,         
        'skewness': skew(clean_z),
        'kurtosis': kurtosis(clean_z),
        
        # -- SENSOR FUSION: 3D Physics --
        # How much did the bike lurch forward/backward?
        'y_axis_variance': np.var(clean_y), 
        'y_axis_range': clean_y.max() - clean_y.min()
    }
    return features

# --- 2. LOAD DATA ---
data = []
labels = []

print("📂 Loading dataset...")

for label in ['speed_breaker', 'normal_road', 'potholes']: 
    label_dir = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_dir):
        continue
        
    files = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
    for filename in files:
        file_path = os.path.join(label_dir, filename)
        try:
            df = pd.read_csv(file_path)
            feats = extract_features(df)
            if feats is not None:
                data.append(feats)
                labels.append(label)
        except Exception as e:
            pass

X = pd.DataFrame(data)
y = np.array(labels)

print(f"✅ Total Samples: {len(X)}")

# --- 3. TRAIN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced', 
    max_depth=10, 
    random_state=42,
    min_samples_leaf=4,         
)
rf.fit(X_train, y_train)

# --- 4. EVALUATE ---
y_pred = rf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(rf, MODEL_NAME)
print(f"💾 Saved Ultimate 13-Feature model to {MODEL_NAME}")