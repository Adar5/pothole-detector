import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from scipy.fft import fft

# Try importing Folium
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(page_title="Dual-Force Detector", layout="wide")
st.title("🛡️ AI Detector (Dual Physics Override)")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists('speed_breaker_model.pkl'):
        return joblib.load('speed_breaker_model.pkl')
    return None

model = load_model()
if model is None:
    st.error("❌ Model not found! Run train_advanced.py")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")

# 1. UNIT CORRECTION
force_convert = st.sidebar.checkbox("✅ Data is in m/s² (Divide by 9.8)", value=True)

# 2. DUAL PHYSICS OVERRIDE (THE FIX)
st.sidebar.divider()
st.sidebar.subheader("💪 Physics Overrides")
# UPDATED: Minimum lowered to 0.5 so you can catch smaller bumps
SB_THRESH = st.sidebar.slider("🔴 Force Breaker (Up G)", 0.5, 3.0, 1.2, help="If signal goes ABOVE this, it's a Speed Breaker.")
PH_THRESH = st.sidebar.slider("🔵 Force Pothole (Down G)", 0.5, 3.0, 0.8, help="If signal drops BELOW this (negative), it's a Pothole.")

# 3. BRAKE FILTER 
st.sidebar.divider()
use_brake_filter = st.sidebar.checkbox("Enable Brake Filter", value=True)
BRAKE_THRESH = st.sidebar.slider("Brake Limit (G)", 0.2, 2.5, 2.0)

# 4. STANDARD SETTINGS
st.sidebar.divider()
FIXED_THRESH = st.sidebar.slider("Minimum Noise Limit (G)", 0.3, 2.0, 0.6) 
CONFIDENCE = st.sidebar.slider("AI Confidence %", 10, 95, 40)
FS_OVERRIDE = st.sidebar.slider("Sampling Rate (Hz)", 10, 200, 100)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- 1. PRE-PROCESSING ---
    sensor_cols = [c for c in df.columns if c.lower().endswith(('x', 'y', 'z')) and 'gyro' not in c.lower()]
    
    if len(sensor_cols) >= 3:
        scale_factor = 9.8 if force_convert else 1.0
        
        means = df[sensor_cols].mean()
        vert_axis_name = means.abs().idxmax()
        gravity_val = means[vert_axis_name]
        horiz_axes = [c for c in sensor_cols if c != vert_axis_name]
        
        # Vertical Signal (Scaled)
        raw_vert = (df[vert_axis_name] - gravity_val) / scale_factor
        if gravity_val < 0: df['signal'] = raw_vert * -1
        else: df['signal'] = raw_vert
            
        # Horizontal Signal (Scaled)
        h_mag = np.sqrt(df[horiz_axes[0]]**2 + df[horiz_axes[1]]**2)
        df['horizontal_g'] = (h_mag - h_mag.mean()) / scale_factor
    else:
        st.error("❌ Need 3-axis accelerometer data.")
        st.stop()

    if 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df)) / FS_OVERRIDE

    # --- 2. SCANNING LOOP ---
    window_len = int(1.0 * FS_OVERRIDE) 
    step_len = int(0.5 * FS_OVERRIDE)
    
    final_events = []
    debug_log = [] 
    
    bar = st.progress(0)
    total_steps = (len(df) - window_len) // step_len
    
    for idx, i in enumerate(range(0, len(df) - window_len, step_len)):
        if idx % 50 == 0 and total_steps > 0: bar.progress(min(idx/total_steps, 1.0))
        
        window = df.iloc[i : i + window_len]
        sig = window['signal']
        horiz = window['horizontal_g']
        
        # Current Window Stats
        max_val = sig.max() # The highest peak
        min_val = sig.min() # The lowest drop (negative)
        abs_max = sig.abs().max()

        # CHECK 1: BRAKE FILTER
        if use_brake_filter and horiz.abs().max() > BRAKE_THRESH:
            continue 

        # CHECK 2: NOISE FILTER
        if abs_max < FIXED_THRESH:
            continue 

        # --- THE FIX: DUAL PHYSICS OVERRIDE ---
        forced_type = None
        
        # Rule A: Big Upward Spike -> Speed Breaker
        if max_val > SB_THRESH:
            forced_type = 'speed_breaker'
            
        # Rule B: Big Downward Drop -> Pothole
        # We check if the drop is deeper than the negative threshold
        elif min_val < (-1 * PH_THRESH): 
            forced_type = 'potholes'

        if forced_type:
             final_events.append({
                'time': window['timestamp'].iloc[0],
                'lat': window['latitude'].mean() if 'latitude' in window.columns else 0,
                'lon': window['longitude'].mean() if 'longitude' in window.columns else 0,
                'type': forced_type, 'confidence': 100, 'intensity': abs_max
            })
             debug_log.append({"Time": window['timestamp'].iloc[0], "Val": round(max_val if forced_type=='speed_breaker' else min_val, 2), "Result": f"💪 FORCE {forced_type}"})
        
        else:
            # ONLY ASK AI IF IT FITS INSIDE THE BRACKETS (Medium Bumps)
            n = len(sig)
            freq_values = fft(sig.values)
            fft_mag = np.abs(freq_values)[:n//2]
            total = np.sum(fft_mag)
            if total == 0: total = 1
            idx_5hz = int(5 * (n / FS_OVERRIDE))
            idx_10hz = int(10 * (n / FS_OVERRIDE))

            features = {
                'max_val': sig.max(), 'min_val': sig.min(),
                'range': sig.max() - sig.min(), 'std_dev': sig.std(),
                'p95': np.percentile(sig.abs(), 95),
                'zero_crossings': (np.diff(np.sign(sig)) != 0).sum(),
                'low_freq_ratio': np.sum(fft_mag[:idx_5hz]) / total,
                'high_freq_ratio': np.sum(fft_mag[idx_10hz:]) / total,
                'dominant_freq_mag': np.max(fft_mag)
            }
            
            try:
                feat_df = pd.DataFrame([features])
                probs = model.predict_proba(feat_df)[0]
                classes = model.classes_
                best_idx = np.argmax(probs)
                pred = classes[best_idx]
                conf = probs[best_idx] * 100
                
                if pred != 'normal_road' and conf >= CONFIDENCE:
                    final_events.append({
                        'time': window['timestamp'].iloc[0],
                        'lat': window['latitude'].mean() if 'latitude' in window.columns else 0,
                        'lon': window['longitude'].mean() if 'longitude' in window.columns else 0,
                        'type': pred, 'confidence': conf, 'intensity': abs_max
                    })
                    debug_log.append({"Time": window['timestamp'].iloc[0], "Val": round(abs_max, 2), "Result": f"✅ AI Found {pred} ({round(conf,0)}%)"})
            except: pass

    bar.progress(100)

    # --- 3. RESULTS ---
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Speed Breakers", sum(1 for e in final_events if e['type'] == 'speed_breaker'))
    c2.metric("🔵 Potholes", sum(1 for e in final_events if 'pothole' in e['type']))
    c3.metric("🔴 Force Up >", f"{SB_THRESH} G")
    c4.metric("🔵 Force Down <", f"-{PH_THRESH} G")

    # GRAPH
    st.subheader("Signal Bracketing")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['signal'], line=dict(color='gray', width=1), name='Vertical (Gs)'))
    
    sb = [e for e in final_events if e['type'] == 'speed_breaker']
    if sb: fig.add_trace(go.Scatter(x=[x['time'] for x in sb], y=[x['intensity'] for x in sb], mode='markers', marker=dict(color='red', size=15), name='Breaker'))
    
    ph = [e for e in final_events if 'pothole' in e['type'] ]
    if ph: fig.add_trace(go.Scatter(x=[x['time'] for x in ph], y=[-(x['intensity']) for x in ph], mode='markers', marker=dict(color='blue', size=12), name='Pothole'))
    
    # Draw Lines
    fig.add_hline(y=SB_THRESH, line_dash="dash", line_color="red", annotation_text="Breaker Zone")
    fig.add_hline(y=-PH_THRESH, line_dash="dash", line_color="blue", annotation_text="Pothole Zone")
    fig.add_hline(y=FIXED_THRESH, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-FIXED_THRESH, line_dash="dot", line_color="green", opacity=0.3)

    st.plotly_chart(fig, use_container_width=True)

    # MAP
    if HAS_FOLIUM and final_events:
        st.subheader("Hazard Map")
        if final_events: start_loc = [final_events[0]['lat'], final_events[0]['lon']]
        else: start_loc = [0,0]
        m = folium.Map(location=start_loc, zoom_start=15)
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satellite').add_to(m)
        if len(df) > 1:
            path = df[['latitude', 'longitude']].iloc[::5].dropna().values.tolist()
            folium.PolyLine(path, color="cyan", weight=4, opacity=0.8).add_to(m)
        for e in final_events:
            color = 'red' if e['type'] == 'speed_breaker' else 'blue'
            folium.Marker([e['lat'], e['lon']], popup=f"{e['type']}", icon=folium.Icon(color=color)).add_to(m)
        st_folium(m, height=400)
    
    st.subheader("Decision Log")
    if debug_log: st.dataframe(pd.DataFrame(debug_log), use_container_width=True)