import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter

st.set_page_config(layout="wide", page_title="Map-Based Data Slicer")
st.title("🗺️ Map-Based Slicer")
st.markdown("1. Use **'Pan Mode'** to move the map. \n2. Switch to **'Select Mode'** to draw a box around the data.")

# --- 1. UPLOAD FILE ---
uploaded_file = st.file_uploader("Upload Raw CSV (Must have latitude/longitude)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Validation
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("❌ This file does not have 'latitude' and 'longitude' columns.")
        st.stop()
        
    # Pre-process Signal
    if 'vibration_only' in df.columns:
        df['signal'] = df['vibration_only']
    else:
        cols = [c for c in df.columns if c.lower().endswith(('x', 'y', 'z')) and 'gyro' not in c.lower()]
        if len(cols) >= 3:
            means = df[cols].mean().abs()
            vert_axis = means.idxmax()
            df['signal'] = df[vert_axis] - df[vert_axis].mean()
        else:
            df['signal'] = 0 

    if 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df)) / 100.0 

    # --- 2. THE INTERACTIVE MAP ---
    st.subheader("1. Locate and Select")
    
    col_map_settings, col_map = st.columns([1, 4])
    
    with col_map_settings:
        st.write("#### Map Controls")
        # FEATURE: Toggle between Panning and Selecting
        map_mode = st.radio("Tool Mode:", ["✋ Pan / Move", "⬜ Box Select"], index=0)
        satellite = st.checkbox("Satellite View", value=False)
        
        drag_mode = 'pan' if map_mode == "✋ Pan / Move" else 'select'

    avg_lat = df['latitude'].mean()
    avg_lon = df['longitude'].mean()
    
    # Downsample just for the visual map
    downsample = 5
    path_df = df.iloc[::downsample].reset_index(drop=True)
    
    with col_map:
        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=path_df['latitude'],
            lon=path_df['longitude'],
            mode='markers+lines', 
            marker=dict(size=6, color='cyan'), # Made dots slightly bigger to catch easier
            line=dict(color='cyan', width=2),
            name='Path'
        ))
        
        fig_map.update_layout(
            mapbox_style="satellite" if satellite else "open-street-map",
            mapbox=dict(center=dict(lat=avg_lat, lon=avg_lon), zoom=16), # Default zoom out slightly
            height=600, # Taller map for easier navigation
            dragmode=drag_mode, # DYNAMIC DRAG MODE
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        selected_map = st.plotly_chart(fig_map, on_select="rerun", use_container_width=True)
    
    # --- 3. FILTER LOGIC ---
    slice_df = pd.DataFrame() 
    
    if selected_map and "selection" in selected_map:
        points = selected_map["selection"]["points"]
        
        if points:
            selected_indices = [p["point_index"] for p in points]
            
            if selected_indices:
                selected_subset = path_df.iloc[selected_indices]
                min_lat = selected_subset['latitude'].min()
                max_lat = selected_subset['latitude'].max()
                min_lon = selected_subset['longitude'].min()
                max_lon = selected_subset['longitude'].max()
                
                mask = (
                    (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
                    (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
                )
                slice_df = df.loc[mask]
                
                st.success(f"✅ Selected Region: {len(slice_df)} data points found inside box.")
    
    # --- 4. GRAPH SELECTION ---
    st.divider()
    
    if slice_df.empty:
        graph_df = df.copy()
        st.caption("Showing Full Drive (Select on Map to zoom in)")
    else:
        graph_df = slice_df.copy()
        st.caption("Showing Map Selection Only")
    
    # Optional Smoothing
    smooth = st.checkbox("Smooth graph signal", value=False)
    if smooth and len(graph_df) > 5:
        graph_df['signal'] = savgol_filter(graph_df['signal'], window_length=min(51, len(graph_df)//2*2+1), polyorder=3)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=graph_df['timestamp'], y=graph_df['signal'], mode='lines', line=dict(color='blue')))
    fig.update_layout(
        title="Vibration Signal",
        xaxis_title="Time (s)",
        yaxis_title="Vibration",
        height=400,
        dragmode='select'
    )
    
    selected_graph = st.plotly_chart(fig, on_select="rerun", use_container_width=True)
    
    final_df = slice_df if not slice_df.empty else pd.DataFrame() # Default to map selection
    
    if selected_graph and "selection" in selected_graph:
        points = selected_graph["selection"]["points"]
        if points:
            indices = [p["point_index"] for p in points]
            final_df = graph_df.iloc[indices]
            st.info(f"Refined Selection: {len(final_df)} points.")
    elif not slice_df.empty:
        final_df = slice_df # Fallback if graph wasn't touched but map was

    # --- 5. DOWNLOAD ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not final_df.empty:
            st.subheader("Preview Selected Clip")
            p_fig = go.Figure()
            p_fig.add_trace(go.Scatter(x=final_df['timestamp'], y=final_df['signal'], mode='lines', line=dict(color='red')))
            p_fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(p_fig, use_container_width=True)
        else:
            st.warning("No data selected.")

    with col2:
        st.subheader("Save")
        if not final_df.empty:
            slice_name = st.text_input("Filename", value="selected_bump_1")
            csv = final_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="⬇️ DOWNLOAD CLIP",
                data=csv,
                file_name=f"{slice_name}.csv",
                mime='text/csv',
                type="primary"
            )