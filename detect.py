import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(page_title="Universal AI Detector", layout="wide")
st.title("🏍️ AI Detector (3D Fusion & Auto-Calibration)")

@st.cache_resource
def load_model():
    if os.path.exists('speed_breaker_model.pkl'):
        return joblib.load('speed_breaker_model.pkl')
    return None

model = load_model()
if model is None:
    st.error("❌ Model not found! Run train_advanced.py")
    st.stop()

# --- THE PHYSICS FIX: BUTTERWORTH FILTER (Restored!) ---
def apply_lowpass_filter(data, cutoff=10, fs=100, order=4):
    """Erases high-frequency engine buzz, leaving only physical bumps."""
    if len(data) <= 15: return data.values if isinstance(data, pd.Series) else data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    try:
        return filtfilt(b, a, data)
    except:
        return data.values if isinstance(data, pd.Series) else data

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")

# 1. UI PROFILES
st.sidebar.divider()
st.sidebar.subheader("🏍️ Vehicle & Mount Profile")
bike_type = st.sidebar.selectbox("Suspension Profile", ["Stiff (Sports/Naked)", "Average (Standard)", "Soft (Scooter/Cruiser)"], index=1)
mount_type = st.sidebar.selectbox("Phone Location", ["Rigid Handlebar Mount", "Jacket Pocket", "Backpack"], index=0)

profile_multiplier = 1.0
if bike_type == "Soft (Scooter/Cruiser)": profile_multiplier *= 1.5
elif bike_type == "Stiff (Sports/Naked)": profile_multiplier *= 0.7

if mount_type == "Jacket Pocket": profile_multiplier *= 1.8
elif mount_type == "Backpack": profile_multiplier *= 2.0

# 2. AUTO-CALIBRATION
st.sidebar.divider()
st.sidebar.subheader("🎛️ Hardware Calibration")
TARGET_STD = st.sidebar.slider("Training Baseline (G)", 0.01, 0.20, 0.05)

# 3. PHYSICS OVERRIDE
st.sidebar.divider()
st.sidebar.subheader("💪 Extreme Overrides")
SB_THRESH = st.sidebar.slider("🔴 Force Breaker (Up G)", 0.5, 3.0, 1.5)
PH_THRESH = st.sidebar.slider("🔵 Force Pothole (Down G)", 0.5, 3.0, 1.2)

# 4. STANDARD FILTERS
st.sidebar.divider()
use_brake_filter = st.sidebar.checkbox("Enable Brake Filter", value=True)
BRAKE_THRESH = st.sidebar.slider("Brake Limit (G)", 0.2, 2.5, 2.0)
FIXED_THRESH = st.sidebar.slider("Minimum Noise Limit (G)", 0.1, 1.0, 0.3) 
CONFIDENCE = st.sidebar.slider("AI Confidence %", 10, 95, 40)
FS_OVERRIDE = st.sidebar.slider("Sampling Rate (Hz)", 10, 200, 100)

# --- FILE UPLOADER (Cleaned up for CSV only) ---
uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Verify the CSV has the exact columns from your screenshot
    if all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
        
        # 1. Process Z-Axis (Vertical)
        gravity_val = df['accel_z'].mean()
        raw_vert = (df['accel_z'] - gravity_val)
        df['raw_signal'] = raw_vert * -1 if gravity_val < 0 else raw_vert
            
        # 2. Process Y-Axis (Forward/Backward Braking)
        df['raw_y'] = df['accel_y'] - df['accel_y'].mean()
            
        # 3. Process Horizontal Magnitude (For the Brake Filter)
        h_mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2)
        df['horizontal_g'] = h_mag - h_mag.mean()
        
        # --- DYNAMIC GAIN CONTROL + UI PROFILES ---
        p5, p95 = df['raw_signal'].quantile(0.05), df['raw_signal'].quantile(0.95)
        live_baseline_std = df['raw_signal'][(df['raw_signal'] > p5) & (df['raw_signal'] < p95)].std()
        if live_baseline_std < 0.01: live_baseline_std = 0.01 
        
        # Combine the auto-calibration with the user's UI Profile choice
        master_multiplier = (TARGET_STD / live_baseline_std) * profile_multiplier
        
        df['signal'] = df['raw_signal'] * master_multiplier
        df['y_signal'] = df['raw_y'] * master_multiplier

        # ---> THE FIX: FILTER THE ENTIRE RIDE ONCE BEFORE THE LOOP <---
        df['clean_z'] = apply_lowpass_filter(df['signal'], fs=FS_OVERRIDE)
        df['clean_y'] = apply_lowpass_filter(df['y_signal'], fs=FS_OVERRIDE)
        
        st.success("✅ Clean CSV Detected: Advanced 3D Swerve & Brake AI Active.")
    else:
        st.error("❌ Missing required columns. Ensure your CSV has 'accel_x', 'accel_y', and 'accel_z'.")
        st.stop()

    # Normalize massive Unix timestamps if they exist
    if 'timestamp' in df.columns:
        if df['timestamp'].max() > 1e11:
            df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1000.0
    else:
        df['timestamp'] = np.arange(len(df)) / FS_OVERRIDE

    # --- UPGRADED SCANNING LOOP (High Overlap & Debouncing) ---
    window_len = int(1.0 * FS_OVERRIDE) 
    step_len = int(0.25 * FS_OVERRIDE) 
    
    final_events = []
    debug_log = [] 
    bar = st.progress(0)
    total_steps = (len(df) - window_len) // step_len
    
    last_detection_time = -999 
    COOLDOWN_SECONDS = 1.5 
    
    for idx, i in enumerate(range(0, len(df) - window_len, step_len)):
        if idx % 50 == 0 and total_steps > 0: bar.progress(min(idx/total_steps, 1.0))
        
        window = df.iloc[i : i + window_len]
        current_time = window['timestamp'].iloc[0]
        
        if (current_time - last_detection_time) < COOLDOWN_SECONDS:
            continue
            
        clean_z = window['clean_z'].values
        clean_y = window['clean_y'].values
        horiz = window['horizontal_g']
        
        max_val = clean_z.max()
        min_val = clean_z.min()
        abs_max = np.abs(clean_z).max()

        if use_brake_filter and horiz.abs().max() > BRAKE_THRESH: continue 
        if abs_max < FIXED_THRESH: continue 

        # The Google Sequence Heuristic (Peak vs Trough)
        max_index = np.argmax(clean_z)
        min_index = np.argmin(clean_z)
        drop_happened_first = min_index < max_index

        forced_type = None
        if max_val > SB_THRESH: forced_type = 'speed_breaker'
        elif min_val < (-1 * PH_THRESH): forced_type = 'potholes'

        pred = None
        conf = 0
        if not forced_type:
            n = len(clean_z)
            freq_values = fft(clean_z)
            fft_mag = np.abs(freq_values)[:n//2]
            total = np.sum(fft_mag) if np.sum(fft_mag) != 0 else 1
            idx_5hz, idx_10hz = int(5 * (n / FS_OVERRIDE)), int(10 * (n / FS_OVERRIDE))

            features = {
                'max_val': clean_z.max(), 'min_val': clean_z.min(),
                'range': clean_z.max() - clean_z.min(), 'std_dev': np.std(clean_z),
                'p95': np.percentile(np.abs(clean_z), 95),
                'zero_crossings': (np.diff(np.sign(clean_z)) != 0).sum(),
                'low_freq_ratio': np.sum(fft_mag[:idx_5hz]) / total,
                'high_freq_ratio': np.sum(fft_mag[idx_10hz:]) / total,
                'dominant_freq_mag': np.max(fft_mag),
                'skewness': skew(clean_z), 'kurtosis': kurtosis(clean_z),
                'y_axis_variance': np.var(clean_y), 'y_axis_range': clean_y.max() - clean_y.min()
            }
            try:
                probs = model.predict_proba(pd.DataFrame([features]))[0]
                best_idx = np.argmax(probs)
                pred = model.classes_[best_idx]
                conf = probs[best_idx] * 100
            except: pass

        # THE EXECUTIVE VETO
        final_type = forced_type if forced_type else pred
        
        if final_type and final_type != 'normal_road' and (forced_type or conf >= CONFIDENCE):
            
            if final_type == 'speed_breaker' and drop_happened_first:
                final_type = 'potholes'
                debug_log.append({"Time": current_time, "Val": round(abs_max, 2), "Result": "🧠 VETO: Rebound fixed to Pothole"})
                
            elif final_type == 'potholes' and not drop_happened_first:
                final_type = 'speed_breaker'
                debug_log.append({"Time": current_time, "Val": round(abs_max, 2), "Result": "🧠 VETO: Shock fixed to Breaker"})
            
            else:
                 debug_log.append({"Time": current_time, "Val": round(abs_max, 2), "Result": f"✅ {final_type} ({round(conf,0)}%)"})

            # Log the event safely handling missing lat/lon
            lat_val = window['latitude'].mean() if 'latitude' in window.columns and not window['latitude'].isna().all() else 0
            lon_val = window['longitude'].mean() if 'longitude' in window.columns and not window['longitude'].isna().all() else 0

            final_events.append({
                'time': current_time,
                'lat': lat_val,
                'lon': lon_val,
                'type': final_type, 'confidence': conf if not forced_type else 100, 'intensity': abs_max
            })
            
            last_detection_time = current_time

    bar.progress(100)

    # --- RESULTS ---
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Speed Breakers", sum(1 for e in final_events if e['type'] == 'speed_breaker'))
    c2.metric("🔵 Potholes", sum(1 for e in final_events if 'pothole' in e['type']))
    c3.metric("Master Multiplier", f"{round(master_multiplier, 2)}x (Auto + UI)")
    c4.metric("AI Confidence", f"> {CONFIDENCE} %")

    # GRAPH
    st.subheader("Filtered & Calibrated Signal (No Engine Noise)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['clean_z'], line=dict(color='gray', width=1), name='Clean Z-Axis'))
    
    sb = [e for e in final_events if e['type'] == 'speed_breaker']
    if sb: fig.add_trace(go.Scatter(x=[x['time'] for x in sb], y=[x['intensity'] for x in sb], mode='markers', marker=dict(color='red', size=15), name='Breaker'))
    
    ph = [e for e in final_events if 'pothole' in e['type'] ]
    if ph: fig.add_trace(go.Scatter(x=[x['time'] for x in ph], y=[-(x['intensity']) for x in ph], mode='markers', marker=dict(color='blue', size=12), name='Pothole'))
    
    fig.add_hline(y=SB_THRESH, line_dash="dash", line_color="red", annotation_text="Force AI Override")
    fig.add_hline(y=-PH_THRESH, line_dash="dash", line_color="blue")

    st.plotly_chart(fig, use_container_width=True)

    # MAP
    if HAS_FOLIUM and 'latitude' in df.columns and 'longitude' in df.columns:
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        if not valid_coords.empty:
            st.subheader("Hazard Map")
            
            if final_events and final_events[0]['lat'] != 0: 
                start_loc = [final_events[0]['lat'], final_events[0]['lon']]
            else: 
                start_loc = [valid_coords['latitude'].iloc[0], valid_coords['longitude'].iloc[0]]
                
            m = folium.Map(location=start_loc, zoom_start=15)
            folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satellite').add_to(m)
            
            if len(valid_coords) > 1:
                path = valid_coords[['latitude', 'longitude']].iloc[::5].values.tolist()
                if path: folium.PolyLine(path, color="cyan", weight=4, opacity=0.8).add_to(m)
                    
            for e in final_events:
                if e['lat'] != 0 and e['lon'] != 0:
                    color = 'red' if e['type'] == 'speed_breaker' else 'blue'
                    folium.Marker([e['lat'], e['lon']], popup=f"{e['type']}", icon=folium.Icon(color=color)).add_to(m)
                
            st_folium(m, height=400)
    
    st.subheader("Decision Log")
    if debug_log: st.dataframe(pd.DataFrame(debug_log), use_container_width=True)