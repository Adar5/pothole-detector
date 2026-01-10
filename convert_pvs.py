import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = 'dataset_gps_mpu_left.csv'
OUTPUT_FILE = 'pvs_cleaned_for_app_100hz.csv'

print(f"Reading {INPUT_FILE}...")

# 1. Load the dataset
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Make sure it is in this folder.")
    exit()

# 2. Select ONLY Dashboard columns (The Passenger Perspective)
keep_cols = [
    'timestamp', 
    'acc_x_dashboard', 'acc_y_dashboard', 'acc_z_dashboard',
    'gyro_x_dashboard', 'gyro_y_dashboard', 'gyro_z_dashboard',
    'latitude', 'longitude', 'speed'
]

# Check if columns exist before proceeding
missing_cols = [c for c in keep_cols if c not in df.columns]
if missing_cols:
    print(f"Error: The file is missing these columns: {missing_cols}")
    print("Are you sure you downloaded 'dataset_gps_mpu_left.csv'?")
    exit()

df_clean = df[keep_cols].copy()

# 3. Rename columns to match your App's expectations
df_clean = df_clean.rename(columns={
    'acc_x_dashboard': 'accel_x',
    'acc_y_dashboard': 'accel_y',
    'acc_z_dashboard': 'accel_z',
    'gyro_x_dashboard': 'gyro_x',
    'gyro_y_dashboard': 'gyro_y',
    'gyro_z_dashboard': 'gyro_z'
})

# 4. FREQUENCY CHECK (No Downsampling)
# Since your phone is 100Hz and this data is 100Hz, we keep it as is.
print(f"Keeping original frequency (100Hz).")
print(f"Total rows: {len(df_clean)}")

# 5. Save
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Success! Created '{OUTPUT_FILE}'")
print("You can now upload this file to your App to extract noise samples.")