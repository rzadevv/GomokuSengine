# pattern_recognition_test.py
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Add parent directory to path to import from root level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GomokuNet
from inference import load_model, predict_move_with_details

# Define board patterns to test
class PatternRecognitionTest:
    def __init__(self, model_path, device='cuda', visualize=True, move_count=8):
        """Initialize the pattern recognition test with the given model."""
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load the model
        self.model = load_model(model_path, device=self.device)
        self.visualize = visualize
        self.move_count = move_count
        self.board_size = 15
        self.correct_count = 0
        self.total_tests = 0
        
        # Results tracking
        self.results = {
            "open_three": {"correct": 0, "total": 0},
            "open_four": {"correct": 0, "total": 0},
            "block_win": {"correct": 0, "total": 0},
            "win_move": {"correct": 0, "total": 0},
            "fork": {"correct": 0, "total": 0},
            "blocked_three": {"correct": 0, "total": 0},
            "blocked_four": {"correct": 0, "total": 0},
            "complex_fork": {"correct": 0, "total": 0},
            "defensive": {"correct": 0, "total": 0},
            "defended_patterns": {"correct": 0, "total": 0},
            "early_game": {"correct": 0, "total": 0}
        }
    
    def create_empty_board(self):
        """Create an empty board in the engine's input format."""
        # Format: (15, 15, 3) where channels are [black, white, empty]
        board = np.zeros((15, 15, 3), dtype=np.float32)
        board[:, :, 2] = 1.0  # All positions start empty
        return board
    
    def place_stone(self, board, position, is_black=True):
        """Place a stone on the board at the given position."""
        row, col = position
        channel = 0 if is_black else 1  # Black = 0, White = 1
        
        # Place the stone
        board[row, col, channel] = 1.0
        board[row, col, 2] = 0.0  # No longer empty
        return board
        
    def get_candidate_moves(self, board, temperature=1.0):
        """Get the candidate moves from the model."""
        _, probs, _, _ = predict_move_with_details(
            self.model, 
            board, 
            temperature=temperature,
            device=self.device
        )
        
        # Get indices of highest probabilities
        flat_probs = probs.flatten()
        selected_indices = np.argsort(flat_probs)[-self.move_count:][::-1]  # Reverse to get descending order
        
        # Convert flat indices to 2D positions
        candidate_moves = []
        for idx in selected_indices:
            row, col = idx // self.board_size, idx % self.board_size
            prob = flat_probs[idx]
            if prob > 0:  # Only consider moves with non-zero probability
                candidate_moves.append(((row, col), prob))
        
        return candidate_moves
    
    def check_move_in_candidates(self, board, expected_move, pattern_name):
        """Check if the expected move is in the candidate moves."""
        self.total_tests += 1
        self.results[pattern_name]["total"] += 1
        
        candidate_moves = self.get_candidate_moves(board)
        candidate_positions = [move[0] for move in candidate_moves]
        
        if expected_move in candidate_positions:
            self.correct_count += 1
            self.results[pattern_name]["correct"] += 1
            result = "PASS"
        else:
            result = "FAIL"
        
        if self.visualize:
            self.visualize_prediction(board, expected_move, candidate_moves, pattern_name, result)
            
        print(f"Test {self.total_tests} ({pattern_name}): {result} - Expected: {expected_move}")
        return expected_move in candidate_positions
    
    def visualize_prediction(self, board, expected_move, candidate_moves, pattern_name, result):
        """Visualize the board, expected move, and candidate predictions."""
        plt.figure(figsize=(10, 8))
        
        # Plot the board
        plt.imshow(np.ones((15, 15, 3)), cmap='Greys')
        
        # Draw grid lines
        for i in range(15):
            plt.axhline(i - 0.5, color='black', linewidth=1)
            plt.axvline(i - 0.5, color='black', linewidth=1)
        
        # Plot stones
        for i in range(15):
            for j in range(15):
                if board[i, j, 0] == 1:  # Black stone
                    plt.plot(j, i, 'o', markersize=15, color='black')
                elif board[i, j, 1] == 1:  # White stone
                    plt.plot(j, i, 'o', markersize=15, color='white', markeredgecolor='black')
        
        # Highlight expected move
        plt.plot(expected_move[1], expected_move[0], 'X', markersize=12, color='red', markeredgewidth=3)
        
        # Highlight candidate moves with consistent size and color
        for ((row, col), prob) in candidate_moves:
            color = 'green' if (row, col) == expected_move else 'blue'
            # Use uniform size and opacity for all candidate moves
            plt.plot(col, row, 'o', markersize=10, 
                     markerfacecolor='none', markeredgewidth=2, 
                     markeredgecolor=color, alpha=0.7)
            
            # Add probability label without ranking
            plt.annotate(f"{prob:.3f}", 
                         xy=(col + 0.3, row), 
                         fontsize=8, color=color)
        
        plt.title(f"Pattern: {pattern_name} - Result: {result}")
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Save figure to parent directory
        results_dir = os.path.join(project_root, "pattern_test_results")
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"test_{self.total_tests}_{pattern_name}_{result}.png"))
        plt.close()
    
    def test_open_three(self):
        """Test recognition of open three patterns."""
        print("\nTesting Open Three Pattern Recognition...")
        
        # Pattern 1: Horizontal open three ○○○
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        # Expected moves to complete: (7,4) or (7,8)
        expected_move = (7, 8)
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Test the other side too
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (7, 4)
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Pattern 2: Vertical open three
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 7), is_black=True)
        board = self.place_stone(board, (6, 7), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (8, 7)
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Pattern 3: Diagonal open three
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 5), is_black=True)
        board = self.place_stone(board, (6, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (8, 8)
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Pattern 4: Anti-diagonal open three
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 9), is_black=True)
        board = self.place_stone(board, (6, 8), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (8, 6)
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Pattern 5: Broken three (○○○)
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        expected_move = (7, 7)  # Fill the gap
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Pattern 6: Double gap three (○○_○)
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (7, 6)  # Fill the gap
        self.check_move_in_candidates(board, expected_move, "open_three")
    
    def test_blocked_three(self):
        """Test recognition of three-in-a-row patterns that are blocked on one side."""
        print("\nTesting Blocked Three Pattern Recognition...")
        
        # Pattern 1: Horizontal three blocked on the right
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=False)  # Blocked by opponent
        expected_move = (7, 4)  # Extend on the open side
        self.check_move_in_candidates(board, expected_move, "blocked_three")
        
        # Pattern 2: Vertical three blocked on top
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 7), is_black=True)
        board = self.place_stone(board, (6, 7), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (4, 7), is_black=False)  # Blocked by opponent
        expected_move = (8, 7)  # Extend on the open side
        self.check_move_in_candidates(board, expected_move, "blocked_three")
        
        # Pattern 3: Diagonal three blocked on one end
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 5), is_black=True)
        board = self.place_stone(board, (6, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=False)  # Blocked by opponent
        expected_move = (4, 4)  # Extend on the open side
        self.check_move_in_candidates(board, expected_move, "blocked_three")
        
        # Pattern 4: Broken three with one end blocked
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        board = self.place_stone(board, (7, 9), is_black=False)  # Blocked by opponent
        expected_move = (7, 7)  # Fill the gap
        self.check_move_in_candidates(board, expected_move, "blocked_three")

    def test_open_four(self):
        """Test recognition of open four patterns."""
        print("\nTesting Open Four Pattern Recognition...")
        
        # Pattern 1: Horizontal open four ○○○○
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (7, 8)  # Win on the right
        self.check_move_in_candidates(board, expected_move, "open_four")
        
        # Test the other side too
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        expected_move = (7, 4)  # Win on the left
        self.check_move_in_candidates(board, expected_move, "open_four")
        
        # Pattern 2: Vertical open four
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 7), is_black=True)
        board = self.place_stone(board, (5, 7), is_black=True)
        board = self.place_stone(board, (6, 7), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (3, 7)  # Win on top
        self.check_move_in_candidates(board, expected_move, "open_four")
        
        # Pattern 3: Diagonal open four
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 4), is_black=True)
        board = self.place_stone(board, (5, 5), is_black=True)
        board = self.place_stone(board, (6, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (3, 3)  # Win on top-left diagonal
        self.check_move_in_candidates(board, expected_move, "open_four")
        
        # Pattern 4: Anti-diagonal open four
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 10), is_black=True)
        board = self.place_stone(board, (5, 9), is_black=True)
        board = self.place_stone(board, (6, 8), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        expected_move = (8, 6)  # Win on bottom-right diagonal
        self.check_move_in_candidates(board, expected_move, "open_four")
        
        # Pattern 5: Broken four (○○○_○)
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        expected_move = (7, 7)  # Fill the gap for win
        self.check_move_in_candidates(board, expected_move, "open_four")
    
    def test_blocked_four(self):
        """Test recognition of four-in-a-row patterns that are blocked on one side."""
        print("\nTesting Blocked Four Pattern Recognition...")
        
        # Pattern 1: Horizontal four blocked on the right
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=False)  # Blocked by opponent
        expected_move = (7, 3)  # Win on the left
        self.check_move_in_candidates(board, expected_move, "blocked_four")
        
        # Pattern 2: Vertical four blocked on top
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 7), is_black=True)
        board = self.place_stone(board, (5, 7), is_black=True)
        board = self.place_stone(board, (6, 7), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (3, 7), is_black=False)  # Blocked by opponent
        expected_move = (8, 7)  # Win on bottom
        self.check_move_in_candidates(board, expected_move, "blocked_four")
        
        # Pattern 3: Diagonal four blocked on one end
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 4), is_black=True)
        board = self.place_stone(board, (5, 5), is_black=True)
        board = self.place_stone(board, (6, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=False)  # Blocked by opponent
        expected_move = (3, 3)  # Win on top-left
        self.check_move_in_candidates(board, expected_move, "blocked_four")
        
        # Pattern 4: Broken four with one end blocked
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        board = self.place_stone(board, (7, 9), is_black=False)  # Blocked by opponent
        expected_move = (7, 7)  # Fill the gap for win
        self.check_move_in_candidates(board, expected_move, "blocked_four")

    def test_block_win(self):
        """Test recognition of blocking opponent's winning moves."""
        print("\nTesting Block Win Pattern Recognition...")
        
        # Pattern 1: Block horizontal win
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=False)  # White stones
        board = self.place_stone(board, (7, 5), is_black=False)
        board = self.place_stone(board, (7, 6), is_black=False)
        board = self.place_stone(board, (7, 7), is_black=False)
        expected_move = (7, 8)  # Block white's win
        self.check_move_in_candidates(board, expected_move, "block_win")
        
        # Pattern 2: Block vertical win
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 7), is_black=False)
        board = self.place_stone(board, (5, 7), is_black=False)
        board = self.place_stone(board, (6, 7), is_black=False)
        board = self.place_stone(board, (7, 7), is_black=False)
        expected_move = (3, 7)  # Block white's win
        self.check_move_in_candidates(board, expected_move, "block_win")
        
        # Pattern 3: Block diagonal win
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 4), is_black=False)
        board = self.place_stone(board, (5, 5), is_black=False)
        board = self.place_stone(board, (6, 6), is_black=False)
        board = self.place_stone(board, (7, 7), is_black=False)
        expected_move = (3, 3)  # Block white's win
        self.check_move_in_candidates(board, expected_move, "block_win")
        
        # Pattern 4: Block broken four
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=False)
        board = self.place_stone(board, (7, 5), is_black=False)
        board = self.place_stone(board, (7, 6), is_black=False)
        board = self.place_stone(board, (7, 8), is_black=False)
        expected_move = (7, 7)  # Block white's win
        self.check_move_in_candidates(board, expected_move, "block_win")
        
        # Pattern 5: Block anti-diagonal win
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 10), is_black=False)
        board = self.place_stone(board, (5, 9), is_black=False)
        board = self.place_stone(board, (6, 8), is_black=False)
        board = self.place_stone(board, (7, 7), is_black=False)
        expected_move = (8, 6)  # Block white's diagonal win
        self.check_move_in_candidates(board, expected_move, "block_win")
        
        # Pattern 6: Block win with own stone nearby
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=False)
        board = self.place_stone(board, (7, 5), is_black=False)
        board = self.place_stone(board, (7, 6), is_black=False)
        board = self.place_stone(board, (7, 7), is_black=False)
        board = self.place_stone(board, (6, 8), is_black=True)  # Own stone nearby
        expected_move = (7, 8)  # Still must block white's win
        self.check_move_in_candidates(board, expected_move, "block_win")
    
    def test_win_move(self):
        """Test recognition of winning moves."""
        print("\nTesting Win Move Pattern Recognition...")
        
        # Pattern 1: Complete horizontal five
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (6, 8), is_black=False)  # Distraction
        expected_move = (7, 8)  # Win
        self.check_move_in_candidates(board, expected_move, "win_move")
        
        # Pattern 2: Complete vertical five
        board = self.create_empty_board()
        board = self.place_stone(board, (3, 7), is_black=True)
        board = self.place_stone(board, (4, 7), is_black=True)
        board = self.place_stone(board, (5, 7), is_black=True)
        board = self.place_stone(board, (6, 7), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=False)  # Distraction
        expected_move = (7, 7)  # Win
        self.check_move_in_candidates(board, expected_move, "win_move")
        
        # Pattern 3: Complete diagonal five
        board = self.create_empty_board()
        board = self.place_stone(board, (4, 4), is_black=True)
        board = self.place_stone(board, (5, 5), is_black=True)
        board = self.place_stone(board, (6, 6), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=True)
        board = self.place_stone(board, (8, 7), is_black=False)  # Distraction
        expected_move = (7, 7)  # Win
        self.check_move_in_candidates(board, expected_move, "win_move")
        
        # Pattern 4: Win with broken four
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        expected_move = (7, 7)  # Win by filling the gap
        self.check_move_in_candidates(board, expected_move, "win_move")
    
    def test_fork(self):
        """Test recognition of fork patterns (creating multiple threats)."""
        print("\nTesting Fork Pattern Recognition...")
        
        # Pattern 1: Double threat fork
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (8, 7), is_black=True)
        board = self.place_stone(board, (9, 7), is_black=True)
        expected_move = (7, 7)  # Creates threats in two directions
        self.check_move_in_candidates(board, expected_move, "fork")
        
        # Pattern 2: Triple threat fork
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (8, 7), is_black=True)
        board = self.place_stone(board, (9, 7), is_black=True)
        board = self.place_stone(board, (6, 8), is_black=True)
        board = self.place_stone(board, (8, 6), is_black=True)
        expected_move = (7, 7)  # Creates threats in multiple directions
        self.check_move_in_candidates(board, expected_move, "fork")
        
        # Pattern 3: L-shaped fork
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (8, 5), is_black=True)
        board = self.place_stone(board, (9, 5), is_black=True)
        expected_move = (7, 7)  # Creates threats in horizontal and vertical
        self.check_move_in_candidates(board, expected_move, "fork")
        
        # Pattern 4: Diamond fork pattern
        board = self.create_empty_board()
        board = self.place_stone(board, (6, 7), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (8, 7), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=True)
        expected_move = (7, 7)  # Creates threats in four directions
        self.check_move_in_candidates(board, expected_move, "fork")
        
        # Pattern 5: Offset fork pattern
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 6), is_black=True)
        board = self.place_stone(board, (6, 5), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 9), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=True)
        expected_move = (7, 7)  # Creates multiple threats with offset pieces
        self.check_move_in_candidates(board, expected_move, "fork")
    
    def test_complex_fork(self):
        """Test recognition of more complex fork patterns."""
        print("\nTesting Complex Fork Pattern Recognition...")
        
        # Pattern 1: Cross-shaped pattern
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 9), is_black=True)
        board = self.place_stone(board, (5, 7), is_black=True)
        board = self.place_stone(board, (9, 7), is_black=True)
        expected_move = (7, 7)  # Center of cross
        self.check_move_in_candidates(board, expected_move, "complex_fork")
        
        # Pattern 2: Diagonal cross pattern
        board = self.create_empty_board()
        board = self.place_stone(board, (5, 5), is_black=True)
        board = self.place_stone(board, (9, 9), is_black=True)
        board = self.place_stone(board, (5, 9), is_black=True)
        board = self.place_stone(board, (9, 5), is_black=True)
        expected_move = (7, 7)  # Center of diagonal cross
        self.check_move_in_candidates(board, expected_move, "complex_fork")
        
        # Pattern 3: Fork with opponent stones nearby
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (8, 7), is_black=True)
        board = self.place_stone(board, (9, 7), is_black=True)
        board = self.place_stone(board, (7, 8), is_black=False)  # Opponent stone
        board = self.place_stone(board, (6, 7), is_black=False)  # Opponent stone
        expected_move = (7, 7)  # Still creates two threats
        self.check_move_in_candidates(board, expected_move, "complex_fork")
        
        # Pattern 4: Multi-directional threat setup
        board = self.create_empty_board()
        board = self.place_stone(board, (6, 6), is_black=True)
        board = self.place_stone(board, (6, 8), is_black=True)
        board = self.place_stone(board, (8, 6), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=True)
        expected_move = (7, 7)  # Center position creating multiple threats
        self.check_move_in_candidates(board, expected_move, "complex_fork")
    
    def test_defended_patterns(self):
        """Test recognition of patterns with opponent stones nearby."""
        print("\nTesting Defended Pattern Recognition...")
        
        # Pattern 1: Defended open three
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (8, 8), is_black=False)  # Opponent nearby
        board = self.place_stone(board, (6, 5), is_black=False)  # Opponent nearby
        expected_move = (7, 8)
        self.check_move_in_candidates(board, expected_move, "open_three")
        
        # Pattern 2: Defended win move
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 4), is_black=True)
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        board = self.place_stone(board, (6, 8), is_black=False)  # Opponent blocking
        board = self.place_stone(board, (8, 8), is_black=False)  # Opponent blocking
        expected_move = (7, 8)  # Still the win move
        self.check_move_in_candidates(board, expected_move, "win_move")
        
        # Pattern 3: Defended against multiple opponents
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=True)
        # Surrounded by opponent stones
        board = self.place_stone(board, (6, 5), is_black=False)
        board = self.place_stone(board, (6, 6), is_black=False)
        board = self.place_stone(board, (6, 7), is_black=False)
        expected_move = (7, 8)  # Still should extend the line
        self.check_move_in_candidates(board, expected_move, "defensive")
        
        # Pattern 4: Defended fork against opponent's block
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)
        board = self.place_stone(board, (8, 7), is_black=True)
        board = self.place_stone(board, (9, 7), is_black=True)
        board = self.place_stone(board, (7, 7), is_black=False)  # Opponent blocking the fork point
        expected_move = (7, 4)  # Create alternative threat
        self.check_move_in_candidates(board, expected_move, "defended_patterns")
        
        # Pattern 5: Counter-threat defense
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=True)
        board = self.place_stone(board, (7, 6), is_black=True)  # Own stones
        board = self.place_stone(board, (5, 7), is_black=False)
        board = self.place_stone(board, (6, 7), is_black=False) 
        board = self.place_stone(board, (7, 7), is_black=False)  # Opponent's three
        expected_move = (7, 4)  # Create own threat while defending
        self.check_move_in_candidates(board, expected_move, "defended_patterns")
    
    def test_defensive(self):
        """Test recognition of defensive patterns."""
        print("\nTesting Defensive Pattern Recognition...")
        
        # Pattern 1: Defend against potential fork
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=False)
        board = self.place_stone(board, (7, 6), is_black=False)
        board = self.place_stone(board, (8, 7), is_black=False)
        board = self.place_stone(board, (9, 7), is_black=False)
        expected_move = (7, 7)  # Prevent opponent's fork
        self.check_move_in_candidates(board, expected_move, "defensive")
        
        # Pattern 2: Defensive move against open three
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=False)
        board = self.place_stone(board, (7, 6), is_black=False)
        board = self.place_stone(board, (7, 7), is_black=False)
        expected_move = (7, 8)  # Block extension on right
        self.check_move_in_candidates(board, expected_move, "defensive")
        
        # Pattern 3: Defend against two separate threats
        board = self.create_empty_board()
        # Two separate groups of opponent stones
        board = self.place_stone(board, (5, 5), is_black=False)
        board = self.place_stone(board, (5, 6), is_black=False)
        board = self.place_stone(board, (5, 7), is_black=False)
        
        board = self.place_stone(board, (7, 8), is_black=False)
        board = self.place_stone(board, (8, 8), is_black=False)
        
        expected_move = (6, 7)  # Block the extension of the horizontal three
        self.check_move_in_candidates(board, expected_move, "defensive")
        
        # Pattern 4: Defend against broken three
        board = self.create_empty_board()
        board = self.place_stone(board, (7, 5), is_black=False)
        board = self.place_stone(board, (7, 6), is_black=False) 
        board = self.place_stone(board, (7, 8), is_black=False)
        expected_move = (7, 7)  # Block the gap in opponent's three
        self.check_move_in_candidates(board, expected_move, "defensive")
    
    def test_early_game(self):
        """Test recognition of early game patterns and opening moves."""
        print("\nTesting Early Game Pattern Recognition...")
        
        # Pattern 1: Empty board - should prefer center or nearby
        board = self.create_empty_board()
        expected_move = (7, 7)  # Center
        self.check_move_in_candidates(board, expected_move, "early_game")

    def run_all_tests(self):
        """Run all pattern recognition tests."""
        self.test_open_three()
        self.test_blocked_three()
        self.test_open_four()
        self.test_blocked_four()
        self.test_block_win()
        self.test_win_move()
        self.test_fork()
        self.test_complex_fork()
        self.test_defensive()
        self.test_defended_patterns()
        self.test_early_game()
        
        # Print summary of results
        print("\n===== Pattern Recognition Test Results =====")
        print(f"Total Tests: {self.total_tests}")
        print(f"Correct Predictions: {self.correct_count}")
        accuracy = 100 * self.correct_count / self.total_tests if self.total_tests > 0 else 0
        print(f"Overall Accuracy: {accuracy:.2f}%")
        
        # Print category-specific results
        print("\nPattern-Specific Results:")
        for pattern, results in self.results.items():
            if results["total"] > 0:
                cat_accuracy = 100 * results["correct"] / results["total"]
                print(f"  {pattern}: {results['correct']}/{results['total']} ({cat_accuracy:.2f}%)")
        
        return accuracy

if __name__ == "__main__":
    parser = ArgumentParser(description="Test Gomoku engine pattern recognition")
    
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser.add_argument("--model", type=str, default=os.path.join(project_root, "best_gomoku_model.pth"), 
                        help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for inference")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualizations for each test")
    parser.add_argument("--move_count", type=int, default=8,
                        help="Number of candidate moves to consider")
    
    args = parser.parse_args()
    
    tester = PatternRecognitionTest(
        model_path=args.model,
        device=args.device,
        visualize=args.visualize,
        move_count=args.move_count
    )
    
    accuracy = tester.run_all_tests()
    
    print(f"\nPattern Recognition :")
    if accuracy >= 90:
        print(f"STRONG ")
    elif accuracy >= 75:
        print(f"MODERATE ")
    else:
        print(f"WEAK ")