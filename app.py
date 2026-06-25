import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os

# Import your custom modules
from src.data.database import save_results, get_history_df
from src.data.ocr import number_extractor
from src.models.gp_model import build_and_sample_gp, predict_next_week

# --- Page Configuration ---
st.set_page_config(page_title="Tactical Board Predictor", layout="wide")
st.title("🎯 Spatiotemporal GP: Target Acquisition")

# --- Helper: Process Single Uploaded Image ---
def process_uploaded_image(uploaded_file):
    """Converts a Streamlit upload into the 20 OCR values"""
    # Convert PIL Image to OpenCV format
    pil_image = Image.open(uploaded_file)
    img_array = np.array(pil_image)
    
    # Ensure RGB
    if len(img_array.shape) == 2: # Grayscale to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # RGBA to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array

    height, width, _ = img_rgb.shape
    rows, cols = 5, 4
    row_step, col_step = height / rows, width / cols
    
    vals = []
    # Create a progress bar for the OCR since it takes a few seconds
    ocr_progress = st.progress(0)
    
    for i in range(20):
        r, c = i // 4, i % 4
        y1, y2 = r * row_step, (r + 1) * row_step
        x1, x2 = c * col_step, (c + 1) * col_step

        tile = img_rgb[int(y1):int(y2), int(x1):int(x2)]
        t_h, t_w, _ = tile.shape
        number_img = tile[int(t_h * 0.5):t_h, 0:t_w]
        
        value = number_extractor(number_img)
        vals.append(value)
        ocr_progress.progress((i + 1) / 20)
        
    return vals

# --- UI Sidebar: Data Ingestion ---
with st.sidebar:
    st.header("1. Upload New Intelligence")
    uploaded_file = st.file_uploader("Upload Previous Week's Board", type=['jpg', 'jpeg', 'png'])
    target_week = st.number_input("Which week is this image for?", min_value=1, value=20, step=1)
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption=f"Week {target_week} Raw Board", use_container_width=True)
        
        if st.button("Process & Save to Database"):
            with st.spinner("Extracting OCR Data..."):
                board_vals = process_uploaded_image(uploaded_file)
                
                # Save to DB (saving a dummy path since it's uploaded via memory)
                save_results(board_vals, f"uploaded_week_{target_week}.jpg", target_week)
                st.success(f"Week {target_week} secured in database!")

# --- UI Main Body: GP Inference ---
st.header("2. Spatiotemporal Inference")

if st.button("Run NUTS Sampler & Generate Targets", type="primary"):
    with st.spinner("Compiling C-Graphs and running MCMC Sampler. This will take a moment..."):
        # 1. Fetch Data
        df = get_history_df()
        df_clean = df.dropna(subset=['reward']).copy()
        
        if df_clean.empty:
            st.error("Database is empty. Please upload data first.")
            st.stop()
            
        df_clean['x_coord'] = df_clean['grid_index'] % 4
        df_clean['y_coord'] = df_clean['grid_index'] // 4
        df_clean['log_reward'] = np.log(df_clean['reward'].astype(float) + 1e-6)

        X_train = df_clean[['x_coord', 'y_coord', 'week_number']].values
        y_train = df_clean['log_reward'].values
        
        target_prediction_week = int(df_clean['week_number'].max() + 1)

        # 2. Run GP (Using the tiny test trace for speed, increase draws/tune for real use)
        model, gp, trace = build_and_sample_gp(X_train, y_train, draws=100, tune=100)
        
        # 3. Predict Next Week
        best_tile, evs, ucbs = predict_next_week(model, gp, trace, target_week=target_prediction_week)

        # --- Render Results ---
        st.success(f"Inference Complete. Projecting for Week {target_prediction_week}.")
        
        # Top-level metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Optimal Target (Tile ID)", value=int(best_tile))
        col2.metric(label="Expected Value (EV)", value=f"{evs[best_tile]:.1f}")
        col3.metric(label="Upper Confidence Bound (UCB)", value=f"{ucbs[best_tile]:.1f}")
        
        st.markdown("---")
        
        # Heatmaps
        col_heat1, col_heat2 = st.columns(2)
        
        # Reshape the 1D arrays (length 20) into 5x4 matrices for Seaborn
        ev_matrix = evs.reshape((5, 4))
        ucb_matrix = ucbs.reshape((5, 4))
        
        with col_heat1:
            st.subheader("Expected Value (Mean Payout)")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(ev_matrix, annot=True, fmt=".1f", cmap="viridis", ax=ax1, cbar_kws={'label': 'Points'})
            ax1.set_xlabel("Grid Column (X)")
            ax1.set_ylabel("Grid Row (Y)")
            st.pyplot(fig1)
            
        with col_heat2:
            st.subheader("Upper Confidence Bound (Optimistic Payout)")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(ucb_matrix, annot=True, fmt=".1f", cmap="magma", ax=ax2, cbar_kws={'label': 'UCB Points'})
            ax2.set_xlabel("Grid Column (X)")
            ax2.set_ylabel("Grid Row (Y)")
            
            # Highlight the best tile mathematically chosen by the UCB
            best_y, best_x = best_tile // 4, best_tile % 4
            ax2.add_patch(plt.Rectangle((best_x, best_y), 1, 1, fill=False, edgecolor='cyan', lw=4))
            
            st.pyplot(fig2)