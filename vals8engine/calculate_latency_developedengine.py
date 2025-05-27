import torch
import numpy as np
import time
import os
import sys
import argparse

# Adjust Python path to import from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes this script is in a subdirectory like vals8engine
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from inference import load_model, predict_move_with_details, place_stone
    # model.py is imported by inference.py, so GomokuNet should be available
except ImportError as e:
    print(f"Error importing necessary modules. Ensure model.py and inference.py are in the project root ({project_root}).")
    print(f"Details: {e}")
    sys.exit(1)

def generate_initial_board(board_size=15):
    """Generates an empty Gomoku board."""
    raw_board = np.zeros((board_size, board_size, 3), dtype=np.float32)
    raw_board[:, :, 2] = 1.0  # All empty
    return raw_board

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Model Loading ---
    print(f"Loading model from: {args.model_path} on device: {device}")
    try:
        model = load_model(args.model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("Model loaded successfully.")

    latencies_ms = []
    current_board = generate_initial_board(args.board_size)
    current_player_is_black = True  # Black starts

    print(f"Performing {args.num_predictions} predictions for latency testing, with board state changes...")

    # Warm-up runs (optional, but good practice)
    if args.warmup_predictions > 0:
        print(f"Performing {args.warmup_predictions} warm-up predictions...")
        for _ in range(args.warmup_predictions):
            predict_move_with_details(
                model, current_board, temperature=0.1, move_count=0,
                device=device, top_k=5, use_value_eval=True
            )
        print("Warm-up complete.")


    for i in range(args.num_predictions):
        start_time = time.perf_counter()
        
        # We use predict_move_with_details as it's often used in main.py/gui.py
        # and represents a comprehensive prediction path.
        # use_value_eval=True ensures the value head evaluation is part of the pipeline.
        prediction_result = predict_move_with_details(
            model,
            current_board,
            temperature=0.1,  # Low temperature for more deterministic (exploitative) moves
            move_count=i,     # Simulate a progressing game state
            device=device,
            top_k=5,          # Consistent with main.py infer mode
            use_value_eval=True
        )
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies_ms.append(latency_ms)
        
        predicted_move_coords = prediction_result[0] # (top_move, probs, initial_value, evaluated_moves)
        
        print(f"Prediction {i+1}/{args.num_predictions}: {latency_ms:.3f} ms (Predicted Move: {predicted_move_coords})")

        # Update board state for the NEXT iteration's prediction
        if predicted_move_coords:
            try:
                current_board = place_stone(current_board, predicted_move_coords, current_player_is_black)
                current_player_is_black = not current_player_is_black
            except ValueError:
                # This might happen if the predicted move is somehow invalid (e.g. on an already occupied cell,
                # though predict_move_with_details should handle masking correctly)
                print(f"Warning: Predicted move {predicted_move_coords} was invalid. Attempting random move to change board state.")
                empty_cells = np.argwhere(current_board[:,:,2] == 1.0)
                if len(empty_cells) > 0:
                    random_idx = np.random.choice(len(empty_cells))
                    r, c = empty_cells[random_idx]
                    current_board = place_stone(current_board, (r+1, c+1), current_player_is_black)
                    current_player_is_black = not current_player_is_black
                else: # Board is full
                    print("Board is full. Resetting board for further latency tests.")
                    current_board = generate_initial_board(args.board_size)
                    current_player_is_black = True
        else:
            # If no move was predicted (e.g., an error within predict_move or board full and it returns None for move)
            print("Warning: No move predicted. Attempting random move or resetting board.")
            empty_cells = np.argwhere(current_board[:,:,2] == 1.0)
            if len(empty_cells) > 0:
                random_idx = np.random.choice(len(empty_cells))
                r, c = empty_cells[random_idx]
                current_board = place_stone(current_board, (r+1, c+1), current_player_is_black)
                current_player_is_black = not current_player_is_black
            else: # Board is full
                print("Board is full. Resetting board.")
                current_board = generate_initial_board(args.board_size)
                current_player_is_black = True
                
    if not latencies_ms:
        print("No latencies recorded. Check prediction loop.")
        return

    average_latency_ms = sum(latencies_ms) / len(latencies_ms)

    print(f"\nTotal predictions: {len(latencies_ms)}")
    print(f"Min latency: {min(latencies_ms):.3f} ms")
    print(f"Max latency: {max(latencies_ms):.3f} ms")
    print(f"Average latency (raw): {average_latency_ms:.3f} ms")
    
    # The user's requested output format:
    print(f"\nIn terms of latency, the developed engine's full move selection averaged {average_latency_ms:.3f} milliseconds per move.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average move selection latency for the Gomoku engine.")
    parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                        help="Path to the trained model checkpoint (relative to project root).")
    parser.add_argument("--num_predictions", type=int, default=50,
                        help="Number of predictions to average for latency calculation.")
    parser.add_argument("--warmup_predictions", type=int, default=5,
                        help="Number of initial predictions to discard as warm-up.")
    parser.add_argument("--board_size", type=int, default=15,
                        help="Size of the Gomoku board.")
    
    cli_args = parser.parse_args()
    
    # Adjust model_path to be relative to project_root if it's not absolute
    if not os.path.isabs(cli_args.model_path):
        cli_args.model_path = os.path.join(project_root, cli_args.model_path)

    main(cli_args) 