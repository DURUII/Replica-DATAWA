
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path (assuming app is running from project root or app folder)
# If running from project root:
sys.path.append(os.path.join(os.getcwd(), 'src'))
# If running from app folder, we might need to go up one level
sys.path.append(os.path.join(os.getcwd(), '../src'))

from preprocess import Preprocessor

st.set_page_config(layout="wide", page_title="Replica Data Visualization")

def get_config():
    return {
        'datasets': [
            {'name': 'Chengdu_Nov01_Paper_Yueche', 'worker': 'CN01_W', 'request': 'CN01_R', 'date': '2016-11-01', 'start_hour': 0, 'end_hour': 3},
            # Add more if needed
        ]
    }

@st.cache_data
def load_data(dataset_idx, grid_mode, delta_t=5):
    config = get_config()
    ds = config['datasets'][dataset_idx]
    
    # Adjust data_dir relative to execution path
    if os.path.exists('data/raw'):
        data_dir = 'data/raw'
    elif os.path.exists('../data/raw'):
        data_dir = '../data/raw'
    else:
        st.error("Data directory not found!")
        return None
    
    if grid_mode == '0.02 Degree (Coarse)':
        preprocessor = Preprocessor(data_dir, grid_size=0.02, grid_size_m=None, delta_t=delta_t)
    else: # 1km
        preprocessor = Preprocessor(data_dir, grid_size_m=1000, delta_t=delta_t)
        
    # This might take a few seconds
    data_dict = preprocessor.process(ds['worker'], ds['request'], ds['start_hour'], ds['end_hour'], ds['date'])
    return data_dict

st.title("🚖 Replica Data Visualization: Heatmap & Sparsity")

# Sidebar
st.sidebar.header("Configuration")

# Dataset Selection
config = get_config()
ds_names = [d['name'] for d in config['datasets']]
ds_idx = st.sidebar.selectbox("Select Dataset", range(len(ds_names)), format_func=lambda x: ds_names[x])

# Grid Mode
grid_mode = st.sidebar.radio("Grid Resolution", ["1km (Fine)", "0.02 Degree (Coarse)"], index=0)

# Load Data
with st.spinner("Loading and Processing Data..."):
    data = load_data(ds_idx, grid_mode)

if data:
    # Extract Data
    C = data['C'] # (T, N, k)
    # Aggregate k to get total demand in that vector (usually 30 mins if k=6, dt=5)
    # Or just sum over k to get "Demand per Vector"
    demand_per_vec = C.sum(axis=2) # (Num_Vectors, Num_Grids)

    min_lat = data['min_lat']
    min_lng = data['min_lng']
    lat_steps, lng_steps = data['grid_shape']
    grid_size_lat = data['grid_size_lat_deg']
    grid_size_lng = data['grid_size_lng_deg']
    start_ts = data['start_ts']
    delta_t = data['delta_t']
    k = data['k']
    vector_duration_min = (delta_t * k) / 60

    # Stats Calculation
    total_cells = lat_steps * lng_steps
    non_zeros_total = (C > 0).sum()
    sparsity_total = non_zeros_total / C.size

    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Statistics")
    st.sidebar.metric("Grid Shape", f"{lat_steps} x {lng_steps}")
    st.sidebar.metric("Total Cells", total_cells)
    st.sidebar.metric("Global Sparsity (Non-zero %)", f"{sparsity_total*100:.2f}%")

    # Time Slider
    num_vectors = demand_per_vec.shape[0]
    base_time = datetime.fromtimestamp(start_ts)

    st.subheader("Time Navigation")
    time_idx = st.slider("Select Time Step", 0, num_vectors-1, 0)
    current_time = base_time + timedelta(minutes=time_idx * vector_duration_min)
    st.write(f"**Current Time:** {current_time} (Duration: {vector_duration_min} mins)")

    # Get Grid for current time
    current_demand = demand_per_vec[time_idx] # (Num_Grids,)
    grid_demand = current_demand.reshape(lat_steps, lng_steps)

    # Current Sparsity
    curr_non_zeros = (current_demand > 0).sum()
    curr_sparsity = curr_non_zeros / total_cells
    st.metric("Current Timestamp Sparsity (Active Cells / Total)", f"{curr_sparsity*100:.2f}%")

    # Visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Map View")
        
        # Calculate Center
        center_lat = min_lat + (lat_steps * grid_size_lat) / 2
        center_lng = min_lng + (lng_steps * grid_size_lng) / 2
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=11, tiles="CartoDB positron")
        
        # Draw Grid Cells
        # We only draw cells with demand > 0 to save performance, or draw all with low opacity
        
        # Find max demand for color scaling
        max_demand = demand_per_vec.max()
        
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap('YlOrRd')
        
        def get_color(val, max_val):
            if val == 0:
                return None
            norm_val = val / max_val if max_val > 0 else 0
            rgba = cmap(norm_val)
            return mcolors.to_hex(rgba)

        # Pre-calculate bounds for all cells to draw outlines? 
        # Drawing 100 rectangles is fine. Drawing 10000 might be slow.
        # 10x10 is 100. 0.02 deg is 5x6=30.
        # Even larger grids (full day) might be bigger.
        # Let's iterate.
        
        for i in range(lat_steps):
            for j in range(lng_steps):
                idx = i * lng_steps + j
                val = current_demand[idx]
                
                # Coords
                # i is lat index (0 is min_lat)
                # j is lng index (0 is min_lng)
                
                # Bottom-Left
                cell_lat_min = min_lat + i * grid_size_lat
                cell_lng_min = min_lng + j * grid_size_lng
                
                # Top-Right
                cell_lat_max = cell_lat_min + grid_size_lat
                cell_lng_max = cell_lng_min + grid_size_lng
                
                color = get_color(val, max_demand)
                
                if color:
                    folium.Rectangle(
                        bounds=[[cell_lat_min, cell_lng_min], [cell_lat_max, cell_lng_max]],
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.6,
                        weight=1,
                        popup=f"Lat: {i}, Lng: {j}, Demand: {val}"
                    ).add_to(m)
                else:
                     # Draw empty grid with light grey
                     folium.Rectangle(
                        bounds=[[cell_lat_min, cell_lng_min], [cell_lat_max, cell_lng_max]],
                        color="#cccccc",
                        weight=0.5,
                        fill=False,
                        opacity=0.3
                    ).add_to(m)

        st_folium(m, width=800, height=500)

    with col2:
        st.subheader("Heatmap Matrix")
        # Show the raw matrix using seaborn/matplotlib
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(grid_demand, ax=ax, cmap="YlOrRd", cbar=True, vmin=0, vmax=max_demand)
        ax.invert_yaxis() # 0 is bottom
        ax.set_title("Grid Demand Matrix")
        st.pyplot(fig)
        
        st.subheader("Temporal Trend")
        # Plot total demand over time
        total_demand_over_time = demand_per_vec.sum(axis=1)
        
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(total_demand_over_time)
        ax2.axvline(x=time_idx, color='r', linestyle='--')
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Total Demand")
        ax2.set_title("Total Network Demand")
        st.pyplot(fig2)
