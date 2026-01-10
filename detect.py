import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# Try to import Folium for the "Google Maps" style
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Speed Breaker Detector AI", layout="wide")
st.title("🤖 AI Speed Breaker Detector (Google Maps Style)")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists('speed_breaker_model.pkl'):
        return joblib.load('speed_breaker_model.pkl')
    else:
        return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Please run 'train_model.py' first.")
    st.stop()

# --- SIDEBAR TUNING ---
st.sidebar.header("🔧 Sensitivity Tuning")
# Defaulted to 5.0 based on your successful test
MIN_VIB_THRESH = st.sidebar.slider("Minimum Vibration (Amplitude)", 0.0, 15.0, 5.0, help="Ignore vibrations smaller than this.")
CONFIDENCE_THRESH = st.sidebar.slider("AI Confidence %", 0, 100, 50, help="Only accept detection if AI is this sure.")
GROUP_SECONDS = st.sidebar.slider("Merge Detections (Seconds)", 0.5, 5.0, 2.0, help="If multiple bumps happen within 2 seconds, count as 1.")

# --- 2. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Driving Data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- PRE-PROCESSING ---
    if 'vibration_only' in df.columns:
        df['signal'] = df['vibration_only']
    else:
        cols = [c for c in df.columns if c.lower().endswith(('x', 'y', 'z'))]
        if len(cols) >= 3:
            # Calculate Magnitude
            df['signal'] = np.sqrt(df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2)
            
            # --- GRAVITY REMOVAL FIX ---
            # Subtract 9.8 so signal centers at 0
            df['signal'] = (df['signal'] - 9.8).abs()
        else:
            st.error("Could not find sensor columns.")
            st.stop()

    if 'timestamp' not in df.columns:
        # Default to 100Hz (0.01s) based on your data
        df['timestamp'] = np.arange(len(df)) / 100.0 

    # --- 3. THE SCANNER LOOP ---
    st.write("🔍 Scanning...")
    progress_bar = st.progress(0)
    
    # Updated to 100Hz based on your data
    fs = 100 
    window_size_sec = 1.0
    step_size_sec = 0.5
    
    window_len = int(window_size_sec * fs)
    step_len = int(step_size_sec * fs)
    
    raw_detections = [] 

    total_steps = (len(df) - window_len) // step_len
    
    # Safe check for empty file
    if total_steps <= 0:
        st.error("File is too short to scan.")
        st.stop()

    for idx, i in enumerate(range(0, len(df) - window_len, step_len)):
        if idx % max(1, (total_steps // 10)) == 0:
            progress_bar.progress(min(idx / total_steps, 1.0))

        window = df.iloc[i : i + window_len]
        signal = window['signal']
        max_val = signal.max()
        
        # --- FILTER 1: AMPLITUDE GATE ---
        if max_val < MIN_VIB_THRESH:
            continue

        # --- EXTRACT FEATURES ---
        features = {
            'max_val': max_val,
            'min_val': signal.min(),
            'range': max_val - signal.min(),
            'std_dev': signal.std(),
            'energy': np.sum(signal**2) / len(signal),
            'p90': np.percentile(signal, 90),
            'kurtosis': pd.Series(signal).kurt()
        }
        feat_df = pd.DataFrame([features])
        
        # --- PREDICT ---
        try:
            probs = model.predict_proba(feat_df)[0]
            if 'speed_breaker' in model.classes_:
                idx_sb = np.where(model.classes_ == 'speed_breaker')[0][0]
                confidence = probs[idx_sb] * 100
            else:
                confidence = 0
            
            if confidence >= CONFIDENCE_THRESH:
                t_stamp = window['timestamp'].iloc[0]
                lat = window['latitude'].mean() if 'latitude' in window.columns else None
                lon = window['longitude'].mean() if 'longitude' in window.columns else None

                raw_detections.append({
                    'time': t_stamp,
                    'lat': lat,
                    'lon': lon,
                    'confidence': confidence,
                    'intensity': max_val
                })
        except: pass

    progress_bar.progress(100)

    # --- FILTER 3: DE-BOUNCING ---
    final_events = []
    if raw_detections:
        current_group = [raw_detections[0]]
        for i in range(1, len(raw_detections)):
            curr = raw_detections[i]
            prev = raw_detections[i-1]
            if (curr['time'] - prev['time']) <= GROUP_SECONDS:
                current_group.append(curr)
            else:
                best_in_group = max(current_group, key=lambda x: x['confidence'])
                final_events.append(best_in_group)
                current_group = [curr]
        if current_group:
            best_in_group = max(current_group, key=lambda x: x['confidence'])
            final_events.append(best_in_group)

    # --- 4. RESULTS ---
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Final Speed Breakers", len(final_events))
    
    # --- GRAPH ---
    st.subheader("1. Filtered Detection Graph")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['signal'], mode='lines', name='Vibration', line=dict(color='lightgray')))
    
    if final_events:
        event_times = [e['time'] for e in final_events]
        fig.add_trace(go.Scatter(
            x=event_times, 
            y=[e['intensity'] for e in final_events],
            mode='markers', name='Confirmed Speed Breaker',
            marker=dict(color='red', size=12, symbol='triangle-down')
        ))
    
    fig.add_hline(y=MIN_VIB_THRESH, line_dash="dash", line_color="green", annotation_text="Min Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- BETTER MAP (FOLIUM) ---
    st.subheader("2. Exact Location Map")
    
    if final_events and final_events[0]['lat'] is not None:
        if HAS_FOLIUM:
            # 1. Create Map centered on the first detection
            avg_lat = np.mean([e['lat'] for e in final_events])
            avg_lon = np.mean([e['lon'] for e in final_events])
            
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15)
            
            # 2. Add Satellite View Option
            folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Satellite',
                overlay = False,
                control = True
            ).add_to(m)

            # 3. Add Red Pins for Speed Breakers
            for e in final_events:
                folium.Marker(
                    [e['lat'], e['lon']],
                    popup=f"<b>Speed Breaker!</b><br>Time: {e['time']:.2f}s<br>Intensity: {e['intensity']:.2f}",
                    icon=folium.Icon(color="red", icon="exclamation-sign")
                ).add_to(m)
            
            # 4. Add User Route (Blue Line)
            # Downsample route for performance (take every 10th point)
            route_coords = df[['latitude', 'longitude']].iloc[::10].dropna().values.tolist()
            if len(route_coords) > 0:
                folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=0.8).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=None, height=500)
            
        else:
            st.warning("⚠️ Please install 'folium' to see the better map: `pip install folium streamlit-folium`")
            st.map(pd.DataFrame(final_events))
    else:
        st.write("No GPS data found or no detections.")