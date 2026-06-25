import numpy as np
from src.data.ocr import process_all_boards
from src.data.database import save_results, get_history_df
from src.models.GP import build_and_train_gp, predict_next_week

def main(run_ingestion=False):
    # --- 1. OCR Data Ingestion ---
    if run_ingestion:
        print("--- Starting OCR Pipeline ---")
        raw_boards = process_all_boards(total_weeks=19)
        
        # Manual OCR Overrides
        if len(raw_boards) >= 15:
            raw_boards[10][1] = 175
            raw_boards[9][0] = 1550
            raw_boards[14][6] = 500

        for week_idx, board in enumerate(raw_boards):
            week_num = week_idx + 1
            img_path = f'data/images/Image{week_num}.jpg'
            save_results(board, img_path, week_num)
    else:
        print("--- Skipping OCR: Loading directly from Database ---")

    # --- 2. Data Preparation ---
    print("\n--- Preparing GP Design Matrix ---")
    df = get_history_df()
    df_clean = df.dropna(subset=['reward']).copy()
    
    df_clean['x_coord'] = df_clean['grid_index'] % 4
    df_clean['y_coord'] = df_clean['grid_index'] // 4
    df_clean['log_reward'] = np.log(df_clean['reward'].astype(float) + 1e-6)

    X_train = df_clean[['x_coord', 'y_coord', 'week_number']].values
    y_train = df_clean['log_reward'].values

    # --- 3. Modeling and Prediction ---
    print("\n--- Executing Spatiotemporal Inference ---")
    target_prediction_week = int(df_clean['week_number'].max() + 1)
    # Increase draws to 1000 when ready for final results
    model, likelihood = build_and_train_gp(X_train, y_train, training_iterations=150)
    best_tile, evs, ucbs = predict_next_week(model, likelihood, target_week=target_prediction_week)
    
    print(f"\n--- Week 20 Tactical Readout ---")
    print(f"Target Acquired: Tile {best_tile} (Column {best_tile % 4}, Row {best_tile // 4})")
    print(f"Expected Base Reward: {evs[best_tile]:.1f}")
    print(f"Upper Confidence Bound: {ucbs[best_tile]:.1f}")

if __name__ == "__main__":
    main()