import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser
import random

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Add path to opponent engine
opponent_path = os.path.join(parent_dir, 'Opponent_gomoku_engine')
sys.path.append(opponent_path)

# Try to import directly from opponent's engine
try:
    from Opponent_gomoku_engine.ai import GomokuAI
    print("Successfully imported opponent's GomokuAI")
except ImportError as e:
    print(f"Error importing opponent's engine: {e}")
    print(f"Make sure the path is correct: {opponent_path}")
    sys.exit(1)

# Define constants for player identifiers
kPlayerBlack = 1
kPlayerWhite = 2
print(f"Using player constants: Black={kPlayerBlack}, White={kPlayerWhite}")

# OpponentPatternTester class
class OpponentPatternTester:
    def __init__(self, model_path=None, visualize=True, move_count=8):
        """Initialize pattern testing for the opponent's engine."""
        # Default model path if none provided
        if model_path is None:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(parent_dir, 'Opponent_gomoku_engine/data/model.pth')
            
        print(f"Initializing Opponent Pattern Testing with model from {model_path}")
        
        self.visualize = visualize
        self.move_count = move_count  # Number of moves to consider
        self.board_size = 15
        self.correct_count = 0
        self.total_tests = 0
        
        # Initialize the opponent's AI
        try:
            print("Initializing opponent's AI...")
            self.ai = GomokuAI()
            
            # Check if the model file exists
            if os.path.exists(model_path):
                print(f"Model found at: {os.path.abspath(model_path)}")
                # Load the model - use the correct method from ConvNet
                self.ai.model.load_model(os.path.basename(model_path))
                print("Successfully loaded opponent's AI model")
            else:
                print(f"WARNING: Model file not found at {model_path}")
                print("Tests will continue using AI without the model")
        except Exception as e:
            print(f"Error initializing opponent's AI: {str(e)}")
            sys.exit(1)
        
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
            "early_game": {"correct": 0, "total": 0},      
            "defended_patterns": {"correct": 0, "total": 0}
        }
    
    def create_empty_board(self):
        """Create an empty board for testing."""
        # We'll use a simpler representation
        board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        return {"board": board, "currentPlayer": kPlayerBlack}
    
    def setup_board(self, stone_positions, black_to_move=True):
        """Set up a board with the given stone positions."""
        # Create a simple board representation
        board = self.create_empty_board()
        
        # Place stones on the board
        for (row, col), is_black in stone_positions:
            player = kPlayerBlack if is_black else kPlayerWhite
            board["board"][row][col] = player
        
        # Set current player
        board["currentPlayer"] = kPlayerBlack if black_to_move else kPlayerWhite
        
        return board
    
    def get_move_options(self, board):
        """Get the move options using the opponent's AI."""
        current_player = board["currentPlayer"]
        
        # Get all valid moves
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board["board"][row][col] == 0:  # Empty position
                    valid_moves.append((row, col))
        
        try:
            # Use GomokuAI directly
            board_array = np.array(board["board"])
            one_hot_board = self.ai.convert_to_one_hot(board_array, current_player)
            
            # Get predictions for all possible moves
            move_options = []
            
            # First, try to get the best move with get_action
            action = self.ai.get_action(board_array, one_hot_board, None)
            
            if action:
                move_options.append((action, 0.9))  # Give highest probability to the AI's chosen move
            
            # Now, let's evaluate each valid move individually to get more options
            remaining_moves = [move for move in valid_moves if move != action]
            move_scores = {}
            
            # Try getting scores for each position
            try:
                # This is an approximation - we're accessing the model directly
                # to get predictions for all board positions
                with torch.no_grad():
                    prediction = self.ai.model(torch.tensor(one_hot_board, dtype=torch.float32).unsqueeze(0))
                    prediction = prediction.squeeze().numpy()
                
                # Get scores for all valid moves from the model's raw output
                for row, col in remaining_moves:
                    move_scores[(row, col)] = prediction[0, row, col]
            except Exception:
                # If we can't get scores from model directly, fall back to distance from existing pieces
                center = self.board_size // 2
                for row, col in remaining_moves:
                    # Score based on distance from center and proximity to existing pieces
                    center_dist = abs(row - center) + abs(col - center)
                    proximity_score = 0
                    
                    # Check proximity to existing pieces
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                if board["board"][nr][nc] != 0:
                                    proximity_score += 10
                    
                    move_scores[(row, col)] = proximity_score - center_dist
            
            # Sort moves by score
            sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
            remaining_prob = 1.0 - (0.9 if action else 0.0)
            priority_moves = sorted_moves[:self.move_count - (1 if action else 0)]
            
            # Normalize scores to get probabilities
            total_score = sum(max(0.01, score) for _, score in priority_moves)
            if total_score > 0:
                for (r, c), score in priority_moves:
                    prob = (max(0.01, score) / total_score) * remaining_prob
                    move_options.append(((r, c), prob))
            
            # Make sure we have enough moves
            if len(move_options) < self.move_count:
                remaining = self.move_count - len(move_options)
                unused_moves = [m for m in valid_moves if m not in [pos for pos, _ in move_options]]
                if unused_moves:
                    more_moves = random.sample(unused_moves, min(remaining, len(unused_moves)))
                    equal_prob = remaining_prob / len(more_moves) if more_moves else 0
                    for move in more_moves:
                        move_options.append((move, equal_prob))
            
            return move_options
        
        except Exception as e:
            print(f"Error using GomokuAI: {str(e)}")
            
            # Fallback: Use center preference plus piece proximity
            move_scores = {}
            center = self.board_size // 2
            
            for row, col in valid_moves:
                # Distance from center (lower is better)
                center_dist = abs(row - center) + abs(col - center)
                score = 100 - (center_dist * 5)  # Base score favoring center
                
                # Add proximity to existing pieces
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            if board["board"][nr][nc] != 0:
                                # Nearby pieces increase score
                                score += 20
                                # Own pieces slightly more
                                if board["board"][nr][nc] == current_player:
                                    score += 5
            
                move_scores[(row, col)] = score
            
            # Sort by score
            sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get moves up to the limit
            priority_moves = sorted_moves[:min(self.move_count, len(sorted_moves))]
            
            # Convert scores to probabilities
            total_score = sum(score for _, score in priority_moves)
            move_options = [(pos, score/total_score) for pos, score in priority_moves]
            
            return move_options
    
    def verify_pattern_move(self, board, expected_move, pattern_name):
        """Check if the expected move matches one of the moves from the AI."""
        self.total_tests += 1
        self.results[pattern_name]["total"] += 1
        
        print(f"\n=== Testing pattern: {pattern_name} ===")
        print(f"Expected move: {expected_move}")
        
        move_options = self.get_move_options(board)
        
        # Convert expected_move to tuple for consistent comparison
        expected_move_tuple = tuple(expected_move) if isinstance(expected_move, list) else expected_move
        
        move_positions = [move[0] for move in move_options]
        
        if expected_move_tuple in move_positions:
            self.correct_count += 1
            self.results[pattern_name]["correct"] += 1
            result = "PASS"
        else:
            result = "FAIL"
        
        if self.visualize:
            self.visualize_prediction(board, expected_move, move_options, pattern_name, result)
        
        print(f"Test {self.total_tests} ({pattern_name}): {result} - Expected move: {expected_move}")
        return expected_move_tuple in move_positions

    def visualize_prediction(self, board, expected_move, move_options, pattern_name, result):
        """Visualize the board, expected move, and model predictions."""
        plt.figure(figsize=(10, 8))
        
        # Plot the board
        plt.imshow(np.ones((self.board_size, self.board_size, 3)), cmap='Greys')
        
        # Draw grid lines
        for i in range(self.board_size):
            plt.axhline(i - 0.5, color='black', linewidth=1)
            plt.axvline(i - 0.5, color='black', linewidth=1)
        
        # Plot stones
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board["board"][row][col] == 1:  # Black stone
                    plt.plot(col, row, 'o', markersize=15, color='black')
                elif board["board"][row][col] == 2:  # White stone
                    plt.plot(col, row, 'o', markersize=15, color='white', markeredgecolor='black')
        
        # Ensure expected_move is a list or tuple
        if not isinstance(expected_move, (list, tuple)):
            expected_move = (expected_move[0], expected_move[1])  # Convert to tuple if needed
            
        # Highlight expected move
        plt.plot(expected_move[1], expected_move[0], 'X', markersize=12, color='red', markeredgewidth=3)
        
        # Convert expected_move to tuple for comparison
        expected_move_tuple = tuple(expected_move) if isinstance(expected_move, list) else expected_move
        
        # Highlight move options without ranking indicators
        for (row, col), prob in move_options:
            position = (row, col)
            color = 'green' if position == expected_move_tuple else 'blue'
            marker_size = 10  # Uniform size for all moves
            
            plt.plot(col, row, 'o', markersize=marker_size, 
                     markerfacecolor='none', markeredgewidth=2, 
                     markeredgecolor=color, alpha=0.7)
            
            # Add probability label
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
        results_dir = os.path.join(project_root, "opponent_pattern_test_results")
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"test_{self.total_tests}_{pattern_name}_{result}.png"))
        plt.close()

    def test_open_three(self):
        """Test recognition of open three patterns."""
        print("\nTesting Open Three Pattern Recognition...")
        
        # Pattern 1: Horizontal open three ○○○
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)
        self.verify_pattern_move(board, expected_move, "open_three")
        
        # Pattern 2: Broken three (○○○)
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap
        self.verify_pattern_move(board, expected_move, "open_three")

    def test_blocked_three(self):
        """Test recognition of three-in-a-row patterns that are blocked on one side."""
        print("\nTesting Blocked Three Pattern Recognition...")
        
        # Pattern 1: Horizontal three blocked on the right
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((7, 8), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 4)  # Extend on the open side
        self.verify_pattern_move(board, expected_move, "blocked_three")
        
        # Pattern 2: Vertical three blocked on top
        stone_positions = [
            ((5, 7), True), ((6, 7), True), ((7, 7), True),
            ((4, 7), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 7)  # Extend on the open side
        self.verify_pattern_move(board, expected_move, "blocked_three")
        
        # Pattern 3: Diagonal three blocked on one end
        stone_positions = [
            ((5, 5), True), ((6, 6), True), ((7, 7), True),
            ((8, 8), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (4, 4)  # Extend on the open side
        self.verify_pattern_move(board, expected_move, "blocked_three")
        
        # Pattern 4: Broken three with one end blocked
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 8), True),
            ((7, 9), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap
        self.verify_pattern_move(board, expected_move, "blocked_three")

    def test_open_four(self):
        """Test recognition of open four patterns."""
        print("\nTesting Open Four Pattern Recognition...")
        
        # Pattern 1: Horizontal open four ○○○○
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Win on the right
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Test the other side too
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 4)  # Win on the left
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 2: Vertical open four
        stone_positions = [
            ((4, 7), True), ((5, 7), True), ((6, 7), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 7)  # Win on top
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 3: Diagonal open four
        stone_positions = [
            ((4, 4), True), ((5, 5), True), ((6, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 3)  # Win on top-left diagonal
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 4: Anti-diagonal open four
        stone_positions = [
            ((4, 10), True), ((5, 9), True), ((6, 8), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 6)  # Win on bottom-right diagonal
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 5: Broken four (○○○_○)
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap for win
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 6: Double broken four (○○_○○)
        stone_positions = [
            ((7, 3), True), ((7, 4), True), ((7, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 5)  # Fill the gap in the middle
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 7: Non-linear four with winning move
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True),
            ((6, 6), True), ((5, 6), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Complete the horizontal open four
        self.verify_pattern_move(board, expected_move, "open_four")
        
        # Pattern 8: Open four with distractions
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((5, 5), False), ((6, 6), False), ((8, 8), False)  # Opponent distractions
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Still complete the open four despite distractions
        self.verify_pattern_move(board, expected_move, "open_four")

    def test_win_move(self):
        """Test recognition of winning moves."""
        print("\nTesting Win Move Pattern Recognition...")
        
        # Pattern 1: Complete horizontal five
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((6, 8), False)  # Distraction
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Win
        self.verify_pattern_move(board, expected_move, "win_move")
        
        # Pattern 2: Complete vertical five
        stone_positions = [
            ((3, 7), True), ((4, 7), True), ((5, 7), True), ((6, 7), True),
            ((8, 8), False)  # Distraction
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Win
        self.verify_pattern_move(board, expected_move, "win_move")

    def test_block_win(self):
        """Test recognition of blocking opponent's winning moves."""
        print("\nTesting Block Win Pattern Recognition...")
        
        # Pattern 1: Block horizontal win
        stone_positions = [
            ((7, 4), False), ((7, 5), False), ((7, 6), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 8)  # Block white's win
        self.verify_pattern_move(board, expected_move, "block_win")
        
        # Pattern 2: Block vertical win
        stone_positions = [
            ((4, 7), False), ((5, 7), False), ((6, 7), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (3, 7)  # Block white's win
        self.verify_pattern_move(board, expected_move, "block_win")
        
        # Pattern 3: Block diagonal win
        stone_positions = [
            ((4, 4), False), ((5, 5), False), ((6, 6), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (3, 3)  # Block white's win
        self.verify_pattern_move(board, expected_move, "block_win")
        
        # Pattern 4: Block broken four
        stone_positions = [
            ((7, 4), False), ((7, 5), False), ((7, 6), False), ((7, 8), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 7)  # Block white's win
        self.verify_pattern_move(board, expected_move, "block_win")
        
        # Pattern 5: Block anti-diagonal win
        stone_positions = [
            ((4, 10), False), ((5, 9), False), ((6, 8), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (8, 6)  # Block white's anti-diagonal win
        self.verify_pattern_move(board, expected_move, "block_win")
        
        # Pattern 6: Block win with own stone nearby
        stone_positions = [
            ((7, 4), False), ((7, 5), False), ((7, 6), False), ((7, 7), False),
            ((6, 8), True)  # Own stone near blocking position
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 8)  # Still need to block white's win despite own stone nearby
        self.verify_pattern_move(board, expected_move, "block_win")
        
        # Pattern 7: Block double-threat win
        stone_positions = [
            ((7, 4), False), ((7, 5), False), ((7, 6), False), ((7, 7), False),  # Horizontal threat
            ((4, 7), False), ((5, 7), False), ((6, 7), False)  # Vertical threat (missing one)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 8)  # Block the immediate win threat (horizontal)
        self.verify_pattern_move(board, expected_move, "block_win")

    def test_fork(self):
        """Test recognition of fork patterns (creating multiple threats)."""
        print("\nTesting Fork Pattern Recognition...")
        
        # Pattern 1: Double threat fork
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Creates threats in two directions
        self.verify_pattern_move(board, expected_move, "fork")
        
        # Pattern 2: Triple threat fork
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True),
            ((6, 8), True), ((8, 6), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Creates threats in multiple directions
        self.verify_pattern_move(board, expected_move, "fork")
        
        # Pattern 3: L-shaped fork
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 5), True), ((9, 5), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Creates threats in horizontal and vertical
        self.verify_pattern_move(board, expected_move, "fork")

        # Pattern 4: Fork with open three
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True),  # Horizontal three
            ((5, 7), True), ((6, 7), True)  # Vertical two
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 7)  # Creates open three and open four simultaneously
        self.verify_pattern_move(board, expected_move, "fork")

    def test_blocked_four(self):
        """Test recognition of four-in-a-row patterns that are blocked on one side."""
        print("\nTesting Blocked Four Pattern Recognition...")
        
        # Pattern 1: Horizontal four blocked on the right
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((7, 8), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 3)  # Win on the left
        self.verify_pattern_move(board, expected_move, "blocked_four")
        
        # Pattern 2: Vertical four blocked on top
        stone_positions = [
            ((4, 7), True), ((5, 7), True), ((6, 7), True), ((7, 7), True),
            ((3, 7), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 7)  # Win on bottom
        self.verify_pattern_move(board, expected_move, "blocked_four")
        
        # Pattern 3: Diagonal four blocked on one end
        stone_positions = [
            ((4, 4), True), ((5, 5), True), ((6, 6), True), ((7, 7), True),
            ((8, 8), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 3)  # Win on top-left
        self.verify_pattern_move(board, expected_move, "blocked_four")
        
        # Pattern 4: Broken four with one end blocked
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 8), True),
            ((7, 9), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap for win
        self.verify_pattern_move(board, expected_move, "blocked_four")
        
        # Pattern 5: Anti-diagonal four blocked on one end
        stone_positions = [
            ((4, 10), True), ((5, 9), True), ((6, 8), True), ((7, 7), True),
            ((3, 11), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 6)  # Win on bottom-right
        self.verify_pattern_move(board, expected_move, "blocked_four")

    def test_complex_fork(self):
        """Test recognition of more complex fork patterns."""
        print("\nTesting Complex Fork Pattern Recognition...")
        
        # Pattern 1: Cross-shaped pattern
        stone_positions = [
            ((7, 5), True), ((7, 9), True), ((5, 7), True), ((9, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Center of cross
        self.verify_pattern_move(board, expected_move, "complex_fork")
        
        # Pattern 2: Diagonal cross pattern
        stone_positions = [
            ((5, 5), True), ((9, 9), True), ((5, 9), True), ((9, 5), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Center of diagonal cross
        self.verify_pattern_move(board, expected_move, "complex_fork")
        
        # Pattern 3: Fork with opponent stones nearby
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True),
            ((7, 8), False), ((6, 7), False)  # Opponent stones
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Still creates two threats
        self.verify_pattern_move(board, expected_move, "complex_fork")
        
        # Pattern 4: Multi-directional threat setup
        stone_positions = [
            ((6, 6), True), ((6, 8), True), ((8, 6), True), ((8, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Center position creating multiple threats
        self.verify_pattern_move(board, expected_move, "complex_fork")
    
    def test_defended_patterns(self):
        """Test recognition of patterns with opponent stones nearby."""
        print("\nTesting Defended Pattern Recognition...")
        
        # Pattern 1: Defended open three
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((8, 8), False), ((6, 5), False)  # Opponent nearby
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)
        self.verify_pattern_move(board, expected_move, "open_three")
        
        # Pattern 2: Defended win move
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((6, 8), False), ((8, 8), False)  # Opponent blocking
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Still the win move
        self.verify_pattern_move(board, expected_move, "win_move")
        
        # Pattern 3: Defended against multiple opponents
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True),
            # Surrounded by opponent stones
            ((6, 5), False), ((6, 6), False), ((6, 7), False)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Still should extend the line
        self.verify_pattern_move(board, expected_move, "defensive")
        
        # Pattern 4: Defended fork against opponent's block
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True),
            ((7, 7), False)  # Opponent blocking the fork point
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 4)  # Create alternative threat
        self.verify_pattern_move(board, expected_move, "defended_patterns")
        
        # Pattern 5: Counter-threat defense
        stone_positions = [
            ((7, 5), True), ((7, 6), True),  # Own stones
            ((5, 7), False), ((6, 7), False), ((7, 7), False)  # Opponent's three
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 4)  # Create own threat while defending
        self.verify_pattern_move(board, expected_move, "defended_patterns")
        
        # Pattern 6: Corner defense formation
        stone_positions = [
            ((7, 7), True), ((8, 8), True), ((6, 6), True),  # Own stones forming pattern
            ((6, 7), False), ((6, 8), False), ((7, 6), False)  # Opponent's surrounding stones
        ]
        board = self.setup_board(stone_positions)
        expected_move = (5, 5)  # Strengthen corner formation
        self.verify_pattern_move(board, expected_move, "defended_patterns")
        
        # Pattern 7: Defense against squeeze play
        stone_positions = [
            ((7, 7), True), ((6, 6), True),  # Own stones
            ((5, 5), False), ((8, 8), False)  # Opponent trying to squeeze
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 6)  # Extend diagonally to defend and attack
        self.verify_pattern_move(board, expected_move, "defended_patterns")
        
        # Pattern 8: Edge defense formation
        stone_positions = [
            ((0, 7), True), ((1, 7), True), ((2, 7), True),  # Own stones at edge
            ((0, 8), False), ((1, 8), False), ((3, 7), False)  # Opponent's surrounding stones
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 8)  # Create defensive shape at edge
        self.verify_pattern_move(board, expected_move, "defended_patterns")

    def test_defensive(self):
        """Test recognition of defensive patterns."""
        print("\nTesting Defensive Pattern Recognition...")
        
        # Pattern 1: Defend against potential fork
        stone_positions = [
            ((7, 5), False), ((7, 6), False), ((8, 7), False), ((9, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 7)  # Prevent opponent's fork
        self.verify_pattern_move(board, expected_move, "defensive")
        
        # Pattern 2: Defensive move against open three
        stone_positions = [
            ((7, 5), False), ((7, 6), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 8)  # Block extension on right
        self.verify_pattern_move(board, expected_move, "defensive")
        
        # Pattern 3: Defend against two separate threats
        stone_positions = [
            # Two separate groups of opponent stones
            ((5, 5), False), ((5, 6), False), ((5, 7), False),
            ((7, 8), False), ((8, 8), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (6, 7)  # Block the extension of the horizontal three
        self.verify_pattern_move(board, expected_move, "defensive")
        
        # Pattern 4: Defend against broken three
        stone_positions = [
            ((7, 5), False), ((7, 6), False), ((7, 8), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 7)  # Block the gap in opponent's broken three
        self.verify_pattern_move(board, expected_move, "defensive")
    
    def test_early_game(self):
        """Test recognition of early game patterns and opening moves."""
        print("\nTesting Early Game Pattern Recognition...")
        
        # Pattern 1: Empty board - should prefer center or nearby
        board = self.create_empty_board()
        expected_move = (7, 7)  # Center
        self.verify_pattern_move(board, expected_move, "early_game")

    def run_all_tests(self):
        """Run all pattern recognition tests."""
        print("\n----- Running Pattern Recognition Tests -----")
        start_time = time.time()
        
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
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n----- Test Results Summary -----")
        print(f"Total Tests: {self.total_tests}")
        print(f"Correct Predictions: {self.correct_count} ({self.correct_count/self.total_tests*100:.1f}%)")
        print(f"Time taken: {duration:.2f} seconds")
        
        # Print results by pattern category
        print("\n----- Results by Pattern Category -----")
        for pattern, data in self.results.items():
            if data["total"] > 0:
                accuracy = data["correct"] / data["total"] * 100
                print(f"{pattern.replace('_', ' ').title()}: {data['correct']}/{data['total']} ({accuracy:.1f}%)")
        
        # Final summary without mentioning rankings
        print("\nPattern Recognition Performance :")
        if self.correct_count / self.total_tests >= 0.9:
            print(".")
        elif self.correct_count / self.total_tests >= 0.7:
            print(".")
        else:
            print(".")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the opponent's model file (default: Opponent_gomoku_engine/data/model.pth)")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize",
                        help="Disable visualization of predictions")
    parser.add_argument("--move_count", type=int, default=8,
                        help="Number of moves to evaluate (default: 8)")
                        
    args = parser.parse_args()
    
    # Create tester instance
    tester = OpponentPatternTester(
        model_path=args.model_path,
        visualize=args.visualize,
        move_count=args.move_count
    )
    
    # Run tests
    print(f"\nPattern Recognition test using opponent's model: {args.model_path}")
    tester.run_all_tests() 