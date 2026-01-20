import pandas as pd
import numpy as np
import os
import joblib
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
MODEL_NAME = 'speed_breaker_model.pkl'

# --- 1. ADVANCED FEATURE EXTRACTION (Frequency Domain) ---
def extract_features(df):
    # 1. Identify Vertical Axis
    axes = ['accel_x', 'accel_y', 'accel_z']
    if 'accel_x' not in df.columns:
        found_cols = [c for c in df.columns if c.lower().endswith(('x', 'y', 'z'))]
        if len(found_cols) >= 3:
            axes = found_cols[:3]
        else:
            return None
            
    means = df[axes].mean()
    vert_axis = means.abs().idxmax()
    
    # 2. Get Signal (Center at 0)
    signal = df[vert_axis] - means[vert_axis]
    
    # 3. TIME DOMAIN FEATURES
    max_val = signal.max()
    min_val = signal.min()
    abs_signal = signal.abs()
    
    # 4. FREQUENCY DOMAIN FEATURES (The Fix)
    # We use FFT to find if the bump is "Low Freq" (Breaker) or "High Freq" (Pothole)
    n = len(signal)
    if n == 0: return None
    
    freq_values = fft(signal.values)
    fft_magnitude = np.abs(freq_values)[:n//2] # Positive frequencies only
    total_energy = np.sum(fft_magnitude)
    
    # Avoid divide by zero
    if total_energy == 0: total_energy = 1
    
    # Calculate energy in Low Freq (0-5Hz) vs High Freq (10Hz+)
    # Assuming 100Hz sampling rate, indices map to freq
    low_freq_energy = np.sum(fft_magnitude[:5]) # Approx 0-10Hz
    high_freq_energy = np.sum(fft_magnitude[10:]) # Approx 20Hz+
    
    features = {
        # -- Time Features --
        'max_val': max_val,
        'min_val': min_val,
        'range': max_val - min_val,
        'std_dev': signal.std(),
        'p95': np.percentile(abs_signal, 95),
        'zero_crossings': (np.diff(np.sign(signal)) != 0).sum(), # How "jittery" is it?
        
        # -- Frequency Features --
        'low_freq_ratio': low_freq_energy / total_energy,   # Higher for Speed Breakers
        'high_freq_ratio': high_freq_energy / total_energy, # Higher for Potholes
        'dominant_freq_mag': np.max(fft_magnitude)          # Strength of main frequency
    }
    return features

# --- 2. LOAD DATA ---
data = []
labels = []

print("📂 Loading dataset...")

for label in ['speed_breaker', 'normal_road', 'potholes']: 
    label_dir = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_dir):
        print(f"❌ Missing folder: {label_dir}")
        continue
        
    files = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
    print(f"   Found {len(files)} samples in {label}...")
    
    for filename in files:
        file_path = os.path.join(label_dir, filename)
        try:
            df = pd.read_csv(file_path)
            feats = extract_features(df)
            if feats:
                data.append(feats)
                labels.append(label)
        except Exception as e:
            pass

X = pd.DataFrame(data)
y = np.array(labels)

print(f"\n✅ Total Samples: {len(X)}")

# --- 3. TRAIN ---
print("🤖 Training with Frequency Analysis...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced', 
    max_depth=10, 
    random_state=42
)
rf.fit(X_train, y_train)

# --- 4. EVALUATE ---
y_pred = rf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

labels_list = ['normal_road', 'speed_breaker', 'potholes']
cm = confusion_matrix(y_test, y_pred, labels=labels_list)
print(f"\nConfusion Matrix (Rows=True, Cols=Pred):\n{cm}")
print(f"Labels: {labels_list}")

joblib.dump(rf, MODEL_NAME)
print(f"\n💾 Saved optimized model to {MODEL_NAME}")