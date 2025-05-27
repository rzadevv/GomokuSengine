import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser
import copy

# Add parent directory to Python path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import correctly from the package structure
try:
    # Try importing from parent directory first
    sys.path.insert(0, os.path.join(parent_dir, 'gomokumcts'))
    from gomokumcts.src.pygomoku.Board import Board
    from gomokumcts.src.pygomoku.Player import PureMCTSPlayer
    from gomokumcts.src.pygomoku.mcts.policy_fn import rollout_policy_fn, MCTS_expand_policy_fn
    print("Successfully imported from gomokumcts package in parent directory")
except ImportError:
    # Otherwise try importing with explicit src structure
    try:
        sys.path.insert(0, os.path.join(current_dir, 'gomokumcts'))
        from gomokumcts.src.pygomoku.Board import Board
        from gomokumcts.src.pygomoku.Player import PureMCTSPlayer
        from gomokumcts.src.pygomoku.mcts.policy_fn import rollout_policy_fn, MCTS_expand_policy_fn
        print("Successfully imported from src.pygomoku")
    except ImportError:
        # Last resort: try to import directly from system path
        try:
            # Also try adding the src directory directly
            src_dir = os.path.join(current_dir, 'gomokumcts', 'src')
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
                
            from gomokumcts.src.pygomoku.Board import Board
            from gomokumcts.src.pygomoku.Player import PureMCTSPlayer
            from gomokumcts.src.pygomoku.mcts.policy_fn import rollout_policy_fn, MCTS_expand_policy_fn
            print("Successfully imported from pygomoku directly")
        except ImportError as e:
            print(f"Import error: {e}")
            print("Please make sure gomokumcts is in your Python path.")
            print(f"Current Python path: {sys.path}")
            sys.exit(1)

# Define board patterns to test
class MCTSPatternRecognitionTest:
    def __init__(self, compute_budget=10000, weight_c=5.0, visualize=True, move_count=8):
        """Initialize the pattern recognition test for MCTS."""
        print(f"Initializing MCTS Pattern Recognition Test with compute_budget={compute_budget}")
        
        self.visualize = visualize
        self.move_count = move_count  # Number of moves to consider
        self.board_size = 15
        self.compute_budget = compute_budget
        self.weight_c = weight_c
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
            "early_game": {"correct": 0, "total": 0}      
        }
    
    def create_empty_board(self):
        """Create an empty Gomoku board."""
        return Board(width=self.board_size, height=self.board_size, numberToWin=5)
    
    def setup_board(self, stone_positions, black_to_move=True):
        """Set up a board with the given stone positions."""
        board = self.create_empty_board()
        
        # Place stones on the board
        for position, is_black in stone_positions:
            row, col = position
            move = board.locationToMove(position)
            # Save the current player
            current_player = board.current_player
            
            # If it's not the right player's turn, undo and redo moves to change players
            if (current_player == Board.kPlayerBlack and not is_black) or \
               (current_player == Board.kPlayerWhite and is_black):
                board.play(0)  # Play a temporary move
                board.undo()   # Undo it to switch player
            
            # Now play the actual move
            board.play(move)
        
        # Make sure it's black's turn if black_to_move is True, otherwise white's turn
        target_player = Board.kPlayerBlack if black_to_move else Board.kPlayerWhite
        if board.current_player != target_player:
            board.play(0)  # Play a temporary move
            board.undo()   # Undo it to switch player
            
        return board
    
    def get_candidate_moves(self, board):
        """Get the candidate moves from MCTS by running multiple searches with noise."""
        # This is a simplified approach - in reality, you might want to 
        # analyze the visit counts of the root's children after a single search
        move_counts = {}
        
        # Create a base MCTS player
        mcts_player = PureMCTSPlayer(
            color=board.current_player,
            weight_c=self.weight_c,
            compute_budget=self.compute_budget // self.move_count,  # Divide budget for multiple runs
            silent=True
        )
        
        print(f"Running multiple MCTS searches to gather move statistics...")
        
        # Run multiple searches
        for i in range(self.move_count):
            # Make a copy of the board to avoid modifying the original
            board_copy = copy.deepcopy(board)
            
            # Reset the MCTS tree for a fresh search
            mcts_player.reset()
            
            # Get move
            move = mcts_player.getAction(board_copy)
            position = board_copy.moveToLocation(move)
            
            # Convert position to tuple to make it hashable for dictionary key
            position_tuple = tuple(position) if position else None
            
            # Count this move
            if position_tuple in move_counts:
                move_counts[position_tuple] += 1
            else:
                move_counts[position_tuple] = 1
            
            # Don't print individual search results to keep output cleaner
        
        # Sort by frequency
        sorted_moves = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Format as [(position, probability)] like in the original
        total_searches = self.move_count
        candidate_moves = [(pos, count/total_searches) for pos, count in sorted_moves]
        
        return candidate_moves
    
    def check_expected_move(self, board, expected_move, pattern_name):
        """Check if the expected move is in the considered moves by the MCTS algorithm."""
        self.total_tests += 1
        self.results[pattern_name]["total"] += 1
        
        candidate_moves = self.get_candidate_moves(board)
        
        # Convert expected_move to tuple for consistent comparison
        expected_move_tuple = tuple(expected_move) if isinstance(expected_move, list) else expected_move
        
        candidate_positions = [move[0] for move in candidate_moves]
        
        if expected_move_tuple in candidate_positions:
            self.correct_count += 1
            self.results[pattern_name]["correct"] += 1
            result = "PASS"
            # Calculate position but don't report it in output
            position = candidate_positions.index(expected_move_tuple) + 1
        else:
            result = "FAIL"
            position = None
        
        if self.visualize:
            self.visualize_prediction(board, expected_move, candidate_moves, pattern_name, result)
            
        print(f"Test {self.total_tests} ({pattern_name}): {result} - Expected move: {expected_move}")
        return expected_move_tuple in candidate_positions
    
    def visualize_prediction(self, board, expected_move, candidate_moves, pattern_name, result):
        """Visualize the board, expected move, and model predictions."""
        plt.figure(figsize=(10, 8))
        
        # Plot the board
        plt.imshow(np.ones((self.board_size, self.board_size, 3)), cmap='Greys')
        
        # Draw grid lines
        for i in range(self.board_size):
            plt.axhline(i - 0.5, color='black', linewidth=1)
            plt.axvline(i - 0.5, color='black', linewidth=1)
        
        # Plot stones
        for move, player in board.states.items():
            row, col = board.moveToLocation(move)
            if player == Board.kPlayerBlack:  # Black stone
                plt.plot(col, row, 'o', markersize=15, color='black')
            else:  # White stone
                plt.plot(col, row, 'o', markersize=15, color='white', markeredgecolor='black')
        
        # Ensure expected_move is a list or tuple
        if not isinstance(expected_move, (list, tuple)):
            expected_move = (expected_move[0], expected_move[1])  # Convert to tuple if needed
            
        # Highlight expected move
        plt.plot(expected_move[1], expected_move[0], 'X', markersize=12, color='red', markeredgewidth=3)
        
        # Convert expected_move to tuple for comparison
        expected_move_tuple = tuple(expected_move) if isinstance(expected_move, list) else expected_move
        
        # Highlight considered moves
        for i, ((row, col), prob) in enumerate(candidate_moves):
            position = (row, col)
            color = 'green' if position == expected_move_tuple else 'blue'
            size = 14 - i  # Decreasing size for lower ranked moves
            alpha = 1.0 - (0.5 * i / len(candidate_moves))  # Higher opacity for higher ranked moves
            
            plt.plot(col, row, 'o', markersize=size, 
                     markerfacecolor='none', markeredgewidth=2, 
                     markeredgecolor=color, alpha=alpha)
            
            # Add probability label without explicitly showing rank
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
        results_dir = os.path.join(project_root, "mcts_pattern_test_results")
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
        self.check_expected_move(board, expected_move, "open_three")
        
        # Test the other side too
        board = self.setup_board(stone_positions)
        expected_move = (7, 4)
        self.check_expected_move(board, expected_move, "open_three")
        
        # Pattern 2: Vertical open three
        stone_positions = [
            ((5, 7), True), ((6, 7), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 7)
        self.check_expected_move(board, expected_move, "open_three")
        
        # Pattern 3: Diagonal open three
        stone_positions = [
            ((5, 5), True), ((6, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 8)
        self.check_expected_move(board, expected_move, "open_three")
        
        # Pattern 4: Anti-diagonal open three
        stone_positions = [
            ((5, 9), True), ((6, 8), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 6)
        self.check_expected_move(board, expected_move, "open_three")
        
        # Pattern 5: Broken three (○○○)
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap
        self.check_expected_move(board, expected_move, "open_three")
        
        # Pattern 6: Double gap three (○○_○)
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 6)  # Fill the gap
        self.check_expected_move(board, expected_move, "open_three")
    
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
        self.check_expected_move(board, expected_move, "blocked_three")
        
        # Pattern 2: Vertical three blocked on top
        stone_positions = [
            ((5, 7), True), ((6, 7), True), ((7, 7), True),
            ((4, 7), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 7)  # Extend on the open side
        self.check_expected_move(board, expected_move, "blocked_three")
        
        # Pattern 3: Diagonal three blocked on one end
        stone_positions = [
            ((5, 5), True), ((6, 6), True), ((7, 7), True),
            ((8, 8), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (4, 4)  # Extend on the open side
        self.check_expected_move(board, expected_move, "blocked_three")
        
        # Pattern 4: Broken three with one end blocked
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 8), True),
            ((7, 9), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap
        self.check_expected_move(board, expected_move, "blocked_three")

    def test_open_four(self):
        """Test recognition of open four patterns."""
        print("\nTesting Open Four Pattern Recognition...")
        
        # Pattern 1: Horizontal open four ○○○○
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Win on the right
        self.check_expected_move(board, expected_move, "open_four")
        
        # Test the other side too
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 4)  # Win on the left
        self.check_expected_move(board, expected_move, "open_four")
        
        # Pattern 2: Vertical open four
        stone_positions = [
            ((4, 7), True), ((5, 7), True), ((6, 7), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 7)  # Win on top
        self.check_expected_move(board, expected_move, "open_four")
        
        # Pattern 3: Diagonal open four
        stone_positions = [
            ((4, 4), True), ((5, 5), True), ((6, 6), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 3)  # Win on top-left diagonal
        self.check_expected_move(board, expected_move, "open_four")
        
        # Pattern 4: Anti-diagonal open four
        stone_positions = [
            ((4, 10), True), ((5, 9), True), ((6, 8), True), ((7, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 6)  # Win on bottom-right diagonal
        self.check_expected_move(board, expected_move, "open_four")
        
        # Pattern 5: Broken four (○○○_○)
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap for win
        self.check_expected_move(board, expected_move, "open_four")
    
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
        self.check_expected_move(board, expected_move, "blocked_four")
        
        # Pattern 2: Vertical four blocked on top
        stone_positions = [
            ((4, 7), True), ((5, 7), True), ((6, 7), True), ((7, 7), True),
            ((3, 7), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (8, 7)  # Win on bottom
        self.check_expected_move(board, expected_move, "blocked_four")
        
        # Pattern 3: Diagonal four blocked on one end
        stone_positions = [
            ((4, 4), True), ((5, 5), True), ((6, 6), True), ((7, 7), True),
            ((8, 8), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (3, 3)  # Win on top-left
        self.check_expected_move(board, expected_move, "blocked_four")
        
        # Pattern 4: Broken four with one end blocked
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 8), True),
            ((7, 9), False)  # Blocked by opponent
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Fill the gap for win
        self.check_expected_move(board, expected_move, "blocked_four")

    def test_block_win(self):
        """Test recognition of blocking opponent's winning moves."""
        print("\nTesting Block Win Pattern Recognition...")
        
        # Pattern 1: Block horizontal win
        stone_positions = [
            ((7, 4), False), ((7, 5), False), ((7, 6), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 8)  # Block white's win
        self.check_expected_move(board, expected_move, "block_win")
        
        # Pattern 2: Block vertical win
        stone_positions = [
            ((4, 7), False), ((5, 7), False), ((6, 7), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (3, 7)  # Block white's win
        self.check_expected_move(board, expected_move, "block_win")
        
        # Pattern 3: Block diagonal win
        stone_positions = [
            ((4, 4), False), ((5, 5), False), ((6, 6), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (3, 3)  # Block white's win
        self.check_expected_move(board, expected_move, "block_win")
        
        # Pattern 4: Block broken four
        stone_positions = [
            ((7, 4), False), ((7, 5), False), ((7, 6), False), ((7, 8), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 7)  # Block white's win
        self.check_expected_move(board, expected_move, "block_win")
    
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
        self.check_expected_move(board, expected_move, "win_move")
        
        # Pattern 2: Complete vertical five
        stone_positions = [
            ((3, 7), True), ((4, 7), True), ((5, 7), True), ((6, 7), True),
            ((8, 8), False)  # Distraction
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Win
        self.check_expected_move(board, expected_move, "win_move")
        
        # Pattern 3: Complete diagonal five
        stone_positions = [
            ((4, 4), True), ((5, 5), True), ((6, 6), True), ((8, 8), True),
            ((8, 7), False)  # Distraction
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Win
        self.check_expected_move(board, expected_move, "win_move")
        
        # Pattern 4: Win with broken four
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Win by filling the gap
        self.check_expected_move(board, expected_move, "win_move")
    
    def test_fork(self):
        """Test recognition of fork patterns (creating multiple threats)."""
        print("\nTesting Fork Pattern Recognition...")
        
        # Pattern 1: Double threat fork
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Creates threats in two directions
        self.check_expected_move(board, expected_move, "fork")
        
        # Pattern 2: Triple threat fork
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True),
            ((6, 8), True), ((8, 6), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Creates threats in multiple directions
        self.check_expected_move(board, expected_move, "fork")
        
        # Pattern 3: L-shaped fork
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 5), True), ((9, 5), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Creates threats in horizontal and vertical
        self.check_expected_move(board, expected_move, "fork")
    
    def test_complex_fork(self):
        """Test recognition of more complex fork patterns."""
        print("\nTesting Complex Fork Pattern Recognition...")
        
        # Pattern 1: Cross-shaped pattern
        stone_positions = [
            ((7, 5), True), ((7, 9), True), ((5, 7), True), ((9, 7), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Center of cross
        self.check_expected_move(board, expected_move, "complex_fork")
        
        # Pattern 2: Diagonal cross pattern
        stone_positions = [
            ((5, 5), True), ((9, 9), True), ((5, 9), True), ((9, 5), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Center of diagonal cross
        self.check_expected_move(board, expected_move, "complex_fork")
        
        # Pattern 3: Fork with opponent stones nearby
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((8, 7), True), ((9, 7), True),
            ((7, 8), False), ((6, 7), False)  # Opponent stones
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Still creates two threats
        self.check_expected_move(board, expected_move, "complex_fork")
        
        # Pattern 4: Multi-directional threat setup
        stone_positions = [
            ((6, 6), True), ((6, 8), True), ((8, 6), True), ((8, 8), True)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 7)  # Center position creating multiple threats
        self.check_expected_move(board, expected_move, "complex_fork")
    
    def test_defended_patterns(self):
        """Test recognition of patterns with opponent stones nearby."""
        print("\nTesting Defended Pattern Recognition...")
        
        # Pattern 1: Defended open three
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((8, 8), False), ((6, 5), False)  # Opponents nearby
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)
        self.check_expected_move(board, expected_move, "open_three")
        
        # Pattern 2: Defended win move
        stone_positions = [
            ((7, 4), True), ((7, 5), True), ((7, 6), True), ((7, 7), True),
            ((6, 8), False), ((8, 8), False)  # Opponents blocking
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Still the win move
        self.check_expected_move(board, expected_move, "win_move")
        
        # Pattern 3: Defended against multiple opponents
        stone_positions = [
            ((7, 5), True), ((7, 6), True), ((7, 7), True),
            # Surrounded by opponent stones
            ((6, 5), False), ((6, 6), False), ((6, 7), False)
        ]
        board = self.setup_board(stone_positions)
        expected_move = (7, 8)  # Still should extend the line
        self.check_expected_move(board, expected_move, "defensive")
    
    def test_defensive(self):
        """Test recognition of defensive patterns."""
        print("\nTesting Defensive Pattern Recognition...")
        
        # Pattern 1: Defend against potential fork
        stone_positions = [
            ((7, 5), False), ((7, 6), False), ((8, 7), False), ((9, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 7)  # Prevent opponent's fork
        self.check_expected_move(board, expected_move, "defensive")
        
        # Pattern 2: Defensive move against open three
        stone_positions = [
            ((7, 5), False), ((7, 6), False), ((7, 7), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (7, 8)  # Block extension on right
        self.check_expected_move(board, expected_move, "defensive")
        
        # Pattern 3: Defend against two separate threats
        stone_positions = [
            # Two separate groups of opponent stones
            ((5, 5), False), ((5, 6), False), ((5, 7), False),
            ((7, 8), False), ((8, 8), False)
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        expected_move = (6, 7)  # Block the extension of the horizontal three
        self.check_expected_move(board, expected_move, "defensive")
    
    def test_early_game(self):
        """Test recognition of early game patterns and opening moves."""
        print("\nTesting Early Game Pattern Recognition...")
        
        # Pattern 1: Empty board - should prefer center or nearby
        board = self.create_empty_board()
        expected_move = (7, 7)  # Center
        self.check_expected_move(board, expected_move, "early_game")
        
        # Pattern 2: Respond to opponent's first move
        stone_positions = [
            ((7, 7), False)  # Opponent in center
        ]
        board = self.setup_board(stone_positions, black_to_move=True)
        # Should play near center
        expected_moves = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 8), (8, 6), (8, 7), (8, 8)]
        # Pick 3 random expected moves to test
        import random
        test_moves = random.sample(expected_moves, 3)
        
        any_correct = False
        for move in test_moves:
            if self.check_expected_move(board, move, "early_game"):
                any_correct = True
        
        if not any_correct:
            # If none of the test moves was in top-k, test one more specific move
            self.check_expected_move(board, (6, 6), "early_game")
        
        # Pattern 3: Third move in the game
        stone_positions = [
            ((7, 7), True),  # First move center
            ((6, 6), False)  # Opponent's response
        ]
        board = self.setup_board(stone_positions)
        
        # Expected moves would be to play near existing stones
        expected_moves = [(5, 5), (5, 6), (5, 7), (6, 5), (6, 7), (6, 8), (7, 6), (7, 8), (8, 6), (8, 7), (8, 8)]
        test_moves = random.sample(expected_moves, 3)
        
        any_correct = False
        for move in test_moves:
            if self.check_expected_move(board, move, "early_game"):
                any_correct = True
        
        if not any_correct:
            # If none of the test moves was in top-k, test one more specific move
            self.check_expected_move(board, (8, 8), "early_game")

    def run_all_tests(self):
        """Run all pattern recognition tests."""
        print("\n----- Running MCTS Pattern Recognition Tests -----")
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
        
        # Final summary without explicit mentions of rankings
        print("\nPattern Recognition Performance :")
        if self.correct_count / self.total_tests >= 0.9:
            print(".")
        elif self.correct_count / self.total_tests >= 0.7:
            print(".")
        else:
            print(".")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--compute_budget", type=int, default=10000,
                        help="Number of simulations for MCTS (default: 10000)")
    parser.add_argument("--weight_c", type=float, default=5.0,
                        help="Exploration weight constant for MCTS (default: 5.0)")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize",
                        help="Disable visualization of predictions")
    parser.add_argument("--move_count", type=int, default=8,
                        help="Number of candidate moves to consider ")
                        
    args = parser.parse_args()
    
    # Create tester instance
    tester = MCTSPatternRecognitionTest(
        compute_budget=args.compute_budget,
        weight_c=args.weight_c,
        visualize=args.visualize,
        move_count=args.move_count
    )
    
    # Run tests
    print(f"\nPattern Recognition test using MCTS with {args.compute_budget} simulations:")
    tester.run_all_tests() 