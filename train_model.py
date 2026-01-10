import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
MODEL_NAME = 'speed_breaker_model.pkl'

# --- 1. FEATURE EXTRACTION ---
def extract_features(df):
    # Auto-detect signal column
    if 'vibration_only' in df.columns:
        signal = df['vibration_only']
    else:
        # Fallback to calculating magnitude
        cols = [c for c in df.columns if c.lower().endswith(('x', 'y', 'z'))]
        if len(cols) >= 3:
            signal = np.sqrt(df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2)
        else:
            return None

    # --- SMARTER FEATURES ---
    # We added 'Kurtosis' and 'P90' to help distinguish bumps from noise
    features = {
        'max_val': signal.max(),
        'min_val': signal.min(),
        'range': signal.max() - signal.min(),
        'std_dev': signal.std(),
        'energy': np.sum(signal**2) / len(signal),
        
        # NEW: 90th percentile (Ignores random single spikes)
        'p90': np.percentile(signal, 90),
        
        # NEW: Kurtosis (Shape of the spike)
        # High kurtosis = Sharp bump (Speed Breaker)
        # Low kurtosis = Constant vibration (Rough Road)
        'kurtosis': df.kurt()['signal'] if 'signal' in df else pd.Series(signal).kurt() 
    }
    return features

# --- 2. LOAD DATA ---
data = []
labels = []

print("📂 Loading dataset...")

for label in ['speed_breaker', 'normal_road']: 
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

# --- 3. TRAIN WITH BALANCE ---
print("🤖 Training Model...")

# Stratify ensures we have both classes in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# KEY FIX: class_weight='balanced'
# This forces the model to pay attention to 'Normal Road' even if there are fewer samples
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# --- 4. EVALUATE ---
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['normal_road', 'speed_breaker'])
print(f"\nConfusion Matrix:\n{cm}")
print("(Top-Right number should be 0. If high, add more Normal Road data.)")

joblib.dump(clf, MODEL_NAME)
print(f"\n💾 Saved {MODEL_NAME}")