import os
import time
import numpy as np
import torch
import torch.nn as nn
import random
import sys
import ai
from ai import GomokuAI
import filereader
import gomoku
from timeit import default_timer as timer
import copy

# Define a model that matches the architecture found in the saved model
class LegacyModel(nn.Module):
    def __init__(self):
        super(LegacyModel, self).__init__()
        # Define layers that match the saved model architecture
        self.conv1 = nn.Conv2d(15, 15, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(225, 225, kernel_size=5, stride=1, padding=2)
        self.layer1 = nn.Sequential(nn.Linear(1, 1))  # Placeholder for layer1
        self.layer2 = nn.Sequential(nn.Linear(1, 1))  # Placeholder for layer2
        self.layer3 = nn.Sequential(nn.Linear(1, 1))  # Placeholder for layer3
        self.fc1 = nn.Linear(1, 1)  # Placeholder
        self.fc2 = nn.Linear(1, 1)  # Placeholder
        self.fc3 = nn.Linear(1, 1)  # Placeholder

    def forward(self, x):
        # We won't actually use this forward pass
        # It's just to match the saved model architecture
        return x

def create_random_board(board_size, fill_factor=0.3):
    """
    Create a random board with pieces placed randomly
    
    Args:
        board_size: Size of the board (e.g., 15 for 15x15)
        fill_factor: Proportion of the board to fill (0-1)
        
    Returns:
        A board with random pieces
    """
    board = [[0] * board_size for _ in range(board_size)]
    total_cells = board_size * board_size
    cells_to_fill = int(total_cells * fill_factor)
    
    filled = 0
    while filled < cells_to_fill:
        row = random.randint(0, board_size-1)
        col = random.randint(0, board_size-1)
        if board[row][col] == 0:
            board[row][col] = random.randint(1, 2)  # 1 or 2 (player pieces)
            filled += 1
            
    return board

def convert_to_one_hot(board, player_id):
    """
    Convert the board to one-hot encoding for the neural network
    """
    board = np.array(board)
    height, width = board.shape
    one_hot_board = np.zeros((3, height, width), dtype=np.float32)
    one_hot_board[0] = (board == 0).astype(np.float32)
    if player_id == 1:
        one_hot_board[1] = (board == 1).astype(np.float32)
        one_hot_board[2] = (board == 2).astype(np.float32)
    else:
        one_hot_board[1] = (board == 2).astype(np.float32)
        one_hot_board[2] = (board == 1).astype(np.float32)
    return one_hot_board

def get_move_without_model(board, player_id):
    """
    Fallback method to get a move without using the neural network model
    This simulates the decision making process of the AI
    """
    # Create a copy of the game values to avoid modifying the original
    values = filereader.create_gomoku_game("consts.json")
    
    # Create a temporary game instance
    temp_instance = gomoku.GomokuGame(values)
    temp_instance.board = copy.deepcopy(board)
    
    # Use the testai to find a move - this will simulate the AI's decision making
    try:
        import testai
        move = testai.ai_move(temp_instance, player_id)
        return move
    except:
        # If testai fails, fall back to random move selection
        valid_moves = []
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == 0:
                    valid_moves.append((row, col))
        
        if not valid_moves:
            return None
        
        return random.choice(valid_moves)

def perform_full_move_selection(board, player_id):
    """
    Simulate the full move selection process as it would happen in the game
    to get an accurate latency measurement
    """
    # Create a copy of the game values
    values = filereader.create_gomoku_game("consts.json")
    
    # Create a temporary game instance 
    temp_instance = gomoku.GomokuGame(values)
    temp_instance.board = copy.deepcopy(board)
    
    # Create temporary player with MM-AI type
    temp_player = gomoku.Player("MM-AI", player_id)
    
    # Get one-hot board representation
    one_hot_board = convert_to_one_hot(board, player_id)
    
    # Calculate scores as in the real game
    try:
        max_score, scores, scores_normalized = gomoku.calculate_score(temp_instance.board)
    except:
        # If score calculation fails, use random scores
        scores_normalized = np.random.random(len(board) * len(board[0])).tolist()
    
    # Get the move using the AI's get_action method
    move = temp_player.ai.get_action(temp_instance.board, one_hot_board, scores_normalized)
    
    return move

def measure_latency(num_predictions=50, board_size=15, warm_up=5, use_real_scoring=False):
    """
    Measure the latency of the move selection process using high precision timing
    
    Args:
        num_predictions: Number of predictions to make
        board_size: Size of the board
        warm_up: Number of warm-up predictions to perform
        use_real_scoring: Whether to use the actual game scoring logic
        
    Returns:
        A dictionary of latency statistics
    """
    # Perform warm-up predictions (silently)
    for _ in range(warm_up):
        board = create_random_board(board_size)
        _ = perform_full_move_selection(board, 1)
    
    # Measure actual predictions
    latencies = []
    predicted_moves = []
    
    for i in range(num_predictions):
        # Create a new random board for each prediction
        # Varying the fill factor to simulate different game stages
        fill_factor = 0.1 + (i / num_predictions) * 0.6  
        board = create_random_board(board_size, fill_factor)
        
        # Measure time for the full move selection process using high precision timer
        start_time = timer()
        move = perform_full_move_selection(board, 1)
        end_time = timer()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        predicted_moves.append(move)
    
    # Calculate statistics
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_latency = sum(latencies) / len(latencies)
    median_latency = sorted(latencies)[len(latencies) // 2]
    
    print(f"Total predictions: {num_predictions}")
    print(f"Min latency: {min_latency:.3f} ms")
    print(f"Max latency: {max_latency:.3f} ms") 
    print(f"Median latency: {median_latency:.3f} ms")
    print(f"Average latency: {avg_latency:.3f} ms")
    
    return {
        "min": min_latency,
        "max": max_latency,
        "avg": avg_latency,
        "median": median_latency,
        "all": latencies
    }

if __name__ == "__main__":
    # Disable print statements from imported modules
    import os
    os.environ['GOMOKU_SILENT'] = '1'
    
    # Suppress CUDA information
    import torch
    torch.set_printoptions(profile="default")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Set the number of predictions to make
    num_predictions = 50
    
    # Measure latency using the full move selection process simulation
    stats = measure_latency(num_predictions=num_predictions) 