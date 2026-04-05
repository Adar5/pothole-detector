import pandas as pd
import numpy as np
import os
import shutil

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
FOLDERS_TO_AUGMENT = ['potholes', 'speed_breaker', 'normal_road']
AUGMENT_FACTOR = 2  # Create 2 fake copies for every 1 real file

print("🚀 Starting Data Augmentation...")

for folder in FOLDERS_TO_AUGMENT:
    folder_path = os.path.join(DATASET_PATH, folder)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping {folder} (Folder not found)")
        continue
        
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'aug' not in f]
    print(f"📂 Processing {folder}: Found {len(files)} original files.")
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(filepath)
            
            # Identify signal columns to modify (don't touch timestamp/GPS)
            # Updated to catch 'x', 'y', 'z', 'vibration', 'signal', etc.
            sensor_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['acc', 'gyro', 'x', 'y', 'z', 'vibration', 'signal'])]
            
            for i in range(1, AUGMENT_FACTOR + 1):
                df_aug = df.copy()
                
                # --- STRATEGY 1: Add Tiny Noise (Simulate different sensor quality) ---
                noise = np.random.normal(0, 0.05, df_aug[sensor_cols].shape)
                df_aug[sensor_cols] = df_aug[sensor_cols] + noise
                
                # --- STRATEGY 2: Scale Intensity (Simulate slightly heavier/lighter car) ---
                scale = np.random.uniform(0.9, 1.1) # +/- 10% intensity
                df_aug[sensor_cols] = df_aug[sensor_cols] * scale
                
                # Save as new file
                new_filename = f"{filename.replace('.csv', '')}_aug_{i}.csv"
                df_aug.to_csv(os.path.join(folder_path, new_filename), index=False)
                
        except Exception as e:
            print(f"❌ Error augmenting {filename}: {e}")

print("✅ Augmentation Complete! Re-run 'train_advanced.py' now.")