#!/usr/bin/env python
import time
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pygomoku.Board import Board
from pygomoku.Player import PureMCTSPlayer
from pygomoku.mcts.policy_fn import rollout_policy_fn, MCTS_expand_policy_fn

def measure_move_latency(num_predictions=50, board_size=15, compute_budget=1000, silent=True):
    """Measure the latency of MCTS move selection"""
    # Create board
    board = Board(width=board_size, height=board_size)
    
    # Create MCTS player
    player = PureMCTSPlayer(
        color=Board.kPlayerBlack, 
        compute_budget=compute_budget, 
        silent=silent
    )
    
    latencies = []
    
    print(f"Starting latency benchmark with {num_predictions} predictions...")
    print(f"Board size: {board_size}x{board_size}, Compute budget: {compute_budget}")
    
    for i in range(num_predictions):
        # Reset for clean state
        board.initBoard()
        player.reset()
        
        # Make some initial moves to create more interesting board positions
        # This simulates being in the middle of a game
        num_initial_moves = np.random.randint(5, 30)
        available_moves = board.availables.copy()
        
        for _ in range(num_initial_moves):
            if not available_moves:
                break
            move_idx = np.random.randint(0, len(available_moves))
            move = available_moves[move_idx]
            board.play(move)
            available_moves = board.availables.copy()
            
            # Check if game ended
            is_end, _ = board.gameEnd()
            if is_end:
                break
        
        # Ensure it's player's turn
        if board.current_player != player.color:
            board.play(np.random.choice(board.availables))
            
        # Measure time to select a move
        start_time = time.time()
        player.getAction(board)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if not silent:
            print(f"Prediction {i+1}/{num_predictions}: {latency_ms:.3f} ms")
    
    # Calculate statistics
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_latency = sum(latencies) / len(latencies)
    
    # Print results
    print("\nLatency Results:")
    print(f"Total predictions: {num_predictions}")
    print(f"Min latency: {min_latency:.3f} ms")
    print(f"Max latency: {max_latency:.3f} ms")
    print(f"Average latency (raw): {avg_latency:.3f} ms")
    
    return latencies

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure MCTS move selection latency')
    parser.add_argument('--predictions', type=int, default=50,
                        help='Number of predictions to measure')
    parser.add_argument('--board_size', type=int, default=15,
                        help='Board size (width and height)')
    parser.add_argument('--compute_budget', type=int, default=1000,
                        help='MCTS compute budget (number of simulations)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information for each prediction')
    
    args = parser.parse_args()
    
    measure_move_latency(
        num_predictions=args.predictions,
        board_size=args.board_size,
        compute_budget=args.compute_budget,
        silent=not args.verbose
    )

if __name__ == "__main__":
    main() 