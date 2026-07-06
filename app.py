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
from src.models.GP import build_and_train_gp, predict_next_week

# --- Page Configuration ---
st.set_page_config(page_title="Tactical Board Predictor", layout="wide")
st.title("🎯 Spatiotemporal GP: Target Acquisition")

# Initialize Session State for Active Play
if 'temp_reveals' not in st.session_state:
    st.session_state.temp_reveals = {}
if 'board_active' not in st.session_state:
    st.session_state.board_active = False

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
    st.info("Only use this when a week is COMPLETELY finished and you want to save it permanently.")
    uploaded_file = st.file_uploader("Upload Finished Board", type=['jpg', 'jpeg', 'png'])
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
st.header("2. Spatiotemporal Inference (Analysis Mode)")

# Start / Reset buttons
col_ctrl1, col_ctrl2 = st.columns([1, 4])
with col_ctrl1:
    if st.button("🚀 Initialize / Run Prediction", type="primary", use_container_width=True):
        st.session_state.board_active = True

with col_ctrl2:
    if st.session_state.board_active and st.session_state.temp_reveals:
        if st.button("🧹 Clear Intermediate Plays & Reset", use_container_width=True):
            st.session_state.temp_reveals = {}
            st.rerun()

st.markdown("---")

if st.session_state.board_active:
    with st.spinner("Conditioning model on historical + intermediate data..."):
        # 1. Fetch Historical Data
        try:
            df = get_history_df()
            df_clean = df.dropna(subset=['reward']).copy()
        except Exception as e:
            st.error(f"Could not load database: {e}")
            st.stop()
            
        if df_clean.empty:
            st.warning("Database is empty. The model will run, but it has no history to learn from.")
            target_prediction_week = 20 # Fallback
        else:
            target_prediction_week = int(df_clean['week_number'].max() + 1)

        # 2. Inject Intermediate Live Plays (The core of Analysis Mode)
        if st.session_state.temp_reveals:
            temp_rows = []
            for t_idx, r_val in st.session_state.temp_reveals.items():
                temp_rows.append({
                    'grid_index': t_idx,
                    'week_number': target_prediction_week,
                    'reward': float(r_val)
                })
            # Combine history with live intermediate session data
            df_clean = pd.concat([df_clean, pd.DataFrame(temp_rows)], ignore_index=True)

        # 3. Preprocess for GP
        df_clean['x_coord'] = df_clean['grid_index'] % 4
        df_clean['y_coord'] = df_clean['grid_index'] // 4
        df_clean['log_reward'] = np.log(df_clean['reward'].astype(float) + 1e-6)

        X_train = df_clean[['x_coord', 'y_coord', 'week_number']].values
        y_train = df_clean['log_reward'].values

        # 4. Run GP Training and Inference
        model, likelihood = build_and_train_gp(X_train, y_train, training_iterations=150)
        best_tile, evs, ucbs = predict_next_week(model, likelihood, target_week=target_prediction_week)
        
        # --- FIX: Clamp known intermediate tiles to their exact values ---
        # Since we know these exact values, there is zero uncertainty.
        if st.session_state.temp_reveals:
            for t_idx, r_val in st.session_state.temp_reveals.items():
                evs[t_idx] = float(r_val)
                ucbs[t_idx] = float(r_val)

        # Determine optimal tile that hasn't been played yet
        available_tiles = [i for i in range(20) if i not in st.session_state.temp_reveals]
        if available_tiles:
            # Overwrite best_tile with the highest UCB among *unplayed* tiles
            best_tile = max(available_tiles, key=lambda idx: ucbs[idx])

        # --- Render Results ---
        st.success(f"Inference Complete. Projecting for Week {target_prediction_week}.")
        
        # Top-level metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Optimal Target (Tile ID)", value=int(best_tile))
        col2.metric(label="Expected Value (EV)", value=f"{evs[best_tile]:.1f}")
        col3.metric(label="Upper Confidence Bound (UCB)", value=f"{ucbs[best_tile]:.1f}")
        
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
            
            # Draw green borders around manually entered intermediate tiles
            for tile_idx in st.session_state.temp_reveals.keys():
                rx, ry = tile_idx % 4, tile_idx // 4
                ax1.add_patch(plt.Rectangle((rx, ry), 1, 1, fill=False, edgecolor='#52b788', lw=4, ls='--'))
                
            st.pyplot(fig1)
            
        with col_heat2:
            st.subheader("Upper Confidence Bound (Optimistic)")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(ucb_matrix, annot=True, fmt=".1f", cmap="magma", ax=ax2, cbar_kws={'label': 'UCB Points'})
            ax2.set_xlabel("Grid Column (X)")
            ax2.set_ylabel("Grid Row (Y)")
            
            # Highlight the UCB target
            best_y, best_x = best_tile // 4, best_tile % 4
            ax2.add_patch(plt.Rectangle((best_x, best_y), 1, 1, fill=False, edgecolor='cyan', lw=4))
            
            # Draw green borders around manually entered intermediate tiles
            for tile_idx in st.session_state.temp_reveals.keys():
                rx, ry = tile_idx % 4, tile_idx // 4
                ax2.add_patch(plt.Rectangle((rx, ry), 1, 1, fill=False, edgecolor='#52b788', lw=4, ls='--'))
                
            st.pyplot(fig2) 
            
        st.markdown("---")
        st.markdown("### 🛠️ Interactive Play Control Desk")
        st.info("Input your mid-week plays here. The board will dynamically re-predict remaining tiles without saving to the database.")
        
        # Input block for adding a new tile
        with st.container(border=True):
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                # Pre-select the recommended tile
                reveal_tile = st.selectbox("Select Played Tile (0-19):", range(20), index=int(best_tile))
            with col_in2:
                reveal_val = st.number_input("Payout Received (Points):", min_value=0, value=100, step=10)
            with col_in3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply Reveal & Re-Predict", use_container_width=True):
                    st.session_state.temp_reveals[reveal_tile] = reveal_val
                    st.rerun()

        # Display currently entered intermediate plays
        if st.session_state.temp_reveals:
            st.write("**Current Intermediate Plays logged this session:**")
            display_dict = {f"Tile {k}": v for k, v in st.session_state.temp_reveals.items()}
            st.json(display_dict)