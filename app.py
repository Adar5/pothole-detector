import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# --- FILTERING FUNCTIONS ---
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# --- APP SETUP ---
st.set_page_config(layout="wide", page_title="Frequency Analyzer")
st.title("🔍 Zoom & Extract Speed Breakers")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    
    st.sidebar.header("1. Map Columns")
    all_cols = df_preview.columns.tolist()
    
    def get_idx(candidates):
        for c in candidates: 
            if c in all_cols: return all_cols.index(c)
        return 0

    x_col = st.sidebar.selectbox("X Axis", all_cols, index=get_idx(['gyro_x', 'accel_x', 'ax']))
    y_col = st.sidebar.selectbox("Y Axis", all_cols, index=get_idx(['gyro_y', 'accel_y', 'ay']))
    z_col = st.sidebar.selectbox("Z Axis", all_cols, index=get_idx(['gyro_z', 'accel_z', 'az']))
    
    st.sidebar.header("2. Filter Settings")
    fs = st.sidebar.number_input("Sampling Rate (Hz)", value=50)
    cutoff = st.sidebar.slider("Cutoff Frequency", 1, 10, 3)

    if st.sidebar.button("Process Data"):
        data = df_preview.copy()
        # Calculate Magnitude
        data['raw_magnitude'] = np.sqrt(data[x_col]**2 + data[y_col]**2 + data[z_col]**2)
        # Apply Filter
        data['vibration_only'] = highpass_filter(data['raw_magnitude'], cutoff, fs)
        # Time Index
        if 'timestamp' not in data.columns:
            data['timestamp'] = np.arange(len(data)) / fs
            
        st.session_state.processed_data = data

    # --- MAIN INTERFACE ---
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        # 1. THE SLIDER (Controls the Graph)
        st.subheader("1. Select & Zoom")
        st.write("Drag the handles to zoom into the graph below.")
        
        t_min = float(data['timestamp'].min())
        t_max = float(data['timestamp'].max())
        
        # The Slider
        start_t, end_t = st.slider(
            "Time Range (Seconds)", 
            min_value=t_min, 
            max_value=t_max, 
            value=(t_min, t_max)
        )
        
        # 2. FILTER DATA FOR PLOTTING (This creates the Zoom effect)
        mask = (data['timestamp'] >= start_t) & (data['timestamp'] <= end_t)
        zoom_data = data.loc[mask]

        # 3. DRAW THE GRAPH (Using only zoomed data)
        fig = go.Figure()
        
        # Raw Data (Grey)
        fig.add_trace(go.Scatter(
            x=zoom_data['timestamp'], y=zoom_data['raw_magnitude'],
            mode='lines', name='Raw Data',
            line=dict(color='lightgrey', width=1)
        ))
        
        # Clean Data (Red)
        fig.add_trace(go.Scatter(
            x=zoom_data['timestamp'], y=zoom_data['vibration_only'],
            mode='lines', name='Clean Signal',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=f"Zoomed View ({start_t:.1f}s - {end_t:.1f}s)",
            xaxis_title="Time (seconds)",
            yaxis_title="Magnitude",
            height=500,
            hovermode="x unified",
            # This ensures the Y-axis scales to fit the CURRENT view, not the whole file
            yaxis=dict(autorange=True) 
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. DOWNLOAD BUTTON
        st.subheader("2. Export This View")
        st.write(f"Selected **{len(zoom_data)}** rows.")
        
        csv = zoom_data.to_csv(index=False)
        st.download_button(
            label="Download Zoomed Data",
            data=csv,
            file_name=f"normal_road_{start_t:.1f}_{end_t:.1f}.csv",
            mime="text/csv"
        )