import numpy as np
import os
import glob
import time
from datetime import datetime
import argparse
import sys
import signal

# Add debug verbosity control
DEBUG_LEVEL = 1  # 0: minimal, 1: normal, 2: detailed, 3: very detailed
USE_PROGRESS_BAR = False  # Global flag for using progress bar
MONITOR_MEMORY = False  # Global flag for memory monitoring
INTERRUPT_REQUESTED = False  # Flag for tracking interrupt requests

# Try importing psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def handle_interrupt(signum, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully"""
    global INTERRUPT_REQUESTED
    if INTERRUPT_REQUESTED:
        debug_print("\nSecond interrupt received! Forcing exit...", 0)
        sys.exit(1)
    else:
        INTERRUPT_REQUESTED = True
        debug_print("\nInterrupt received! Processing will stop after the current file finishes.", 0)
        debug_print("Press Ctrl+C again to force immediate exit.", 0)

# Register the interrupt handler
signal.signal(signal.SIGINT, handle_interrupt)

def set_debug_level(level):
    """Set the global debug level"""
    global DEBUG_LEVEL
    DEBUG_LEVEL = level

def set_use_progress_bar(use_bar):
    """Set whether to use progress bar"""
    global USE_PROGRESS_BAR
    USE_PROGRESS_BAR = use_bar

def set_monitor_memory(monitor):
    """Set whether to monitor memory usage"""
    global MONITOR_MEMORY
    if monitor and not PSUTIL_AVAILABLE:
        print("Warning: psutil not installed, memory monitoring disabled")
        MONITOR_MEMORY = False
    else:
        MONITOR_MEMORY = monitor

def get_memory_usage():
    """Get current memory usage in MB"""
    if not PSUTIL_AVAILABLE:
        return 0
    
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        ram_usage = mem_info.rss / 1024 / 1024  # Convert bytes to MB
        
        # Add GPU memory tracking if available and using CUDA
        gpu_usage = 0
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_available():
                try:
                    # Get current GPU memory allocated by PyTorch
                    gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                    # Include cached memory which may be reused
                    gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
                    return ram_usage + gpu_usage  # Return combined memory usage
                except Exception as e:
                    debug_print(f"Error tracking GPU memory: {e}", 1)
                    # Continue with RAM only if GPU tracking fails
        
        return ram_usage
    except Exception as e:
        debug_print(f"Error measuring memory usage: {e}", 1)
        return 0  # Return 0 on error to avoid crashing

def debug_print(message, level=1):
    """Print debug messages based on verbosity level"""
    if DEBUG_LEVEL >= level:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end='\r'):
    """
    Call in a loop to create a terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\n") (Str)
    """
    if not USE_PROGRESS_BAR:
        return
        
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print a new line on completion
    if iteration == total: 
        print()

def parse_psq_file(filepath):
    """
    Parse a .psq file to extract the board size, moves, and winner information.
    
    Expected file format:
        - The first line is a header (e.g., "Piskvorky 20x20, 11:11, 0")
          from which the board size is extracted.
        - Subsequent lines with three comma-separated values (e.g., "3,17,23457")
          represent moves. The third value is assumed to be a time (in ms).
        - The moves section stops when a non-move line is encountered (e.g., engine references,
          a marker "-1", or any line that cannot be parsed into three integers).
          
    Args:
        filepath (str): Path to the .psq file.
        
    Returns:
        board_size (int): The size of the board (e.g., 20 for a 20x20 board).
        moves (list of tuples): Each tuple is (x, y, time) where x and y are 1-indexed.
        winner (int): 1 if black wins, -1 if white wins, 0 if draw. None if undetermined.
    """
    moves = []
    board_size = None
    winner = None
    parse_errors = 0
    start_time = time.time()
    
    debug_print(f"Parsing file: {filepath}", 2)
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            debug_print(f"Error: Empty file - {filepath}", 1)
            return None, [], None
            
        # Process header to extract board size (assumes header like "Piskvorky 20x20, 11:11, 0")
        header = lines[0].strip()
        debug_print(f"Header: {header}", 3)
        header_parts = header.split(',')
        if header_parts:
            game_info = header_parts[0]  # e.g., "Piskvorky 20x20"
            tokens = game_info.split()
            board_size_parsed = False
            for token in tokens:
                if 'x' in token:
                    dims = token.split('x')
                    try:
                        if len(dims) == 2 and dims[0].isdigit() and dims[1].isdigit():
                            board_size = int(dims[0])
                            board_size_parsed = True
                            debug_print(f"Board size parsed: {board_size}x{board_size}", 3)
                    except ValueError:
                        pass
                        
            if not board_size_parsed:
                debug_print(f"Warning: Could not parse board size from header '{header}' in {filepath}", 1)
                
        # Default to 15 if board_size was not parsed
        if board_size is None:
            board_size = 15
            debug_print(f"Warning: Using default board size 15 for {filepath}", 1)
        
        # Validate board size is reasonable
        if board_size < 5 or board_size > 30:
            debug_print(f"Error: Unusual board size {board_size} in {filepath}, skipping file", 1)
            return None, [], None

        # Process subsequent lines to extract moves
        move_count = 0
        
        for line_num, line in enumerate(lines[1:], 2):
            line = line.strip()
            # Stop if we encounter a termination marker or a non-move line
            if line == "-1":
                debug_print(f"Found termination marker at line {line_num}", 3)
                break
                
            # Expecting exactly two commas (three parts)
            if line.count(',') != 2:
                parse_errors += 1
                debug_print(f"Parse error at line {line_num}, invalid format: {line}", 3)
                continue
                
            parts = line.split(',')
            try:
                # Try converting each part to integer
                x = int(parts[0])
                y = int(parts[1])
                time_val = int(parts[2])
                
                # Check for out-of-bounds moves (with margin for error)
                if x < 1 or x > board_size + 2 or y < 1 or y > board_size + 2:
                    debug_print(f"Warning: Out-of-bounds move ({x},{y}) in {filepath} line {line_num}", 2)
                    parse_errors += 1
                    continue
                    
                moves.append((x, y, time_val))
                move_count += 1
            except ValueError:
                # If conversion fails, skip this line (likely engine meta-data)
                parse_errors += 1
                debug_print(f"Parse error at line {line_num}, cannot convert to int: {line}", 3)
                continue
        
        # File integrity check
        if parse_errors > move_count * 0.1 and parse_errors > 3:  # More than 10% errors and at least 3 errors
            debug_print(f"Warning: High error rate in {filepath}: {parse_errors} errors in {move_count + parse_errors} lines", 1)
            
        # Verify we have a reasonable number of moves
        if move_count < 5:
            debug_print(f"Warning: Too few moves ({move_count}) in {filepath}, possibly corrupted", 1)
            
        # Determine the winner based on the game record
        if moves:
            debug_print(f"Determining winner from {len(moves)} moves", 3)
            winner = determine_winner(moves, board_size)
            if winner:
                debug_print(f"Winner determined: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", 3)
            else:
                debug_print(f"Winner undetermined", 3)
            
    except Exception as e:
        debug_print(f"Error parsing file {filepath}: {str(e)}", 1)
        import traceback
        traceback.print_exc()
        return None, [], None
    
    elapsed = time.time() - start_time
    debug_print(f"Parsed {move_count} moves in {elapsed:.2f}s with {parse_errors} errors", 2)
    
    return board_size, moves, winner

def determine_winner(moves, board_size):
    """
    Determine the winner of a game based on the move list.
    
    In Gomoku, a player wins by forming an unbroken chain of 5 or more stones horizontally,
    vertically, or diagonally. This function uses exact win checking for every move.
    
    Args:
        moves (list of tuples): List of moves as (x, y, time).
        board_size (int): Size of the board.
        
    Returns:
        int: 1 if black wins, -1 if white wins, 0 if draw. None if undetermined.
    """
    if not moves:
        return None  # Empty move list, cannot determine winner
        
    # Initialize an empty board
    board = np.zeros((board_size, board_size), dtype=np.int8)
    
    # Fill the board with moves (1 for black, -1 for white)
    current_player = 1  # Black starts
    
    for idx, (x, y, _) in enumerate(moves):
        # Convert to 0-indexed
        x_idx, y_idx = x - 1, y - 1
        
        # Skip invalid moves
        if not (0 <= x_idx < board_size and 0 <= y_idx < board_size):
            continue
            
        if board[x_idx, y_idx] != 0:
            continue
            
        # Place the stone
        board[x_idx, y_idx] = current_player
        
        # Check if this move is a winning move using exact checking
        if check_win(board, x_idx, y_idx, current_player):
            return current_player
            
        # Switch player
        current_player = -current_player
    
    # If we've gone through all moves and no one has won, it's a draw
    # Only declare a draw if the board is mostly filled (80%+)
    filled_positions = np.sum(board != 0)
    if filled_positions >= board_size * board_size * 0.8:
        return 0
    
    # For incomplete games, return None instead of guessing
    return None

def check_win(board, x, y, player):
    """
    Check if the last move at (x, y) resulted in a win.
    Uses exact win condition checking for the standard 5-in-a-row rule.
    In Standard Gomoku, only exactly 5 in a row is a win, overlines are not winning.
    
    Args:
        board (np.array): The game board.
        x (int): The x-coordinate of the last move (0-indexed).
        y (int): The y-coordinate of the last move (0-indexed).
        player (int): The player who made the move (1 for black, -1 for white).
        
    Returns:
        bool: True if the move resulted in a win, False otherwise.
    """
    board_size = board.shape[0]
    
    # Directions: horizontal, vertical, diagonal (\), diagonal (/)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        # Count stones in a row
        count = 1  # Start with 1 for the current position
        stones_in_line = [(x, y)]  # Track positions to check for overlines
        
        # Check in both directions along the current line
        for direction in [1, -1]:
            for i in range(1, 5):  # Look up to 4 stones away in each direction
                nx, ny = x + direction * i * dx, y + direction * i * dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == player:
                        count += 1
                        stones_in_line.append((nx, ny))
                    else:
                        break
                else:
                    break
                    
        # Check if exactly 5 in a row
        if count == 5:
            # Now check if there's an overline (more than 5)
            is_overline = False
            
            # Find the extremes of the 5-in-a-row
            stones_in_line.sort()  # Sort by x, then by y
            min_x, min_y = stones_in_line[0]
            max_x, max_y = stones_in_line[-1]
            
            # Check one position beyond each end for additional stones
            if dx == 1 and dy == 0:  # Horizontal
                # Check left
                if min_x > 0 and board[min_x-1, min_y] == player:
                    is_overline = True
                # Check right
                if max_x < board_size-1 and board[max_x+1, max_y] == player:
                    is_overline = True
                    
            elif dx == 0 and dy == 1:  # Vertical
                # Check top
                if min_y > 0 and board[min_x, min_y-1] == player:
                    is_overline = True
                # Check bottom
                if max_y < board_size-1 and board[max_x, max_y+1] == player:
                    is_overline = True
                    
            elif dx == 1 and dy == 1:  # Diagonal (\)
                # Check top-left
                if min_x > 0 and min_y > 0 and board[min_x-1, min_y-1] == player:
                    is_overline = True
                # Check bottom-right
                if max_x < board_size-1 and max_y < board_size-1 and board[max_x+1, max_y+1] == player:
                    is_overline = True
                    
            elif dx == 1 and dy == -1:  # Diagonal (/)
                # Find actual extremes for this direction
                top_right = max(stones_in_line, key=lambda pos: pos[0] - pos[1])
                bottom_left = min(stones_in_line, key=lambda pos: pos[0] - pos[1])
                
                # Check top-right
                tr_x, tr_y = top_right
                if tr_x < board_size-1 and tr_y > 0 and board[tr_x+1, tr_y-1] == player:
                    is_overline = True
                # Check bottom-left
                bl_x, bl_y = bottom_left
                if bl_x > 0 and bl_y < board_size-1 and board[bl_x-1, bl_y+1] == player:
                    is_overline = True
            
            # Return True only if it's exactly 5 in a row, not an overline
            if not is_overline:
                return True
            
    return False

def center_crop(array, output_size):
    """
    Center-crop a 2D or 3D array to the given output size.
    
    Args:
        array (np.ndarray): Input array with shape (H, W) or (H, W, C).
        output_size (int): Desired height and width after crop.
        
    Returns:
        np.ndarray: Cropped array.
    """
    h, w = array.shape[:2]
    start_x = (h - output_size) // 2
    start_y = (w - output_size) // 2
    if array.ndim == 3:
        return array[start_x:start_x+output_size, start_y:start_y+output_size, :]
    else:
        return array[start_x:start_x+output_size, start_y:start_y+output_size]

def generate_training_examples(moves, board_size, winner, output_size=15,
                               opening_threshold=10, midgame_threshold=20):
    """
    Generate training examples from the list of moves.
    
    For each move (except the last), the function:
      - Updates the board state (using a 3-channel representation)
      - Validates moves and skips examples from corrupted sequences
      - Constructs the input board (3-channel) and the target (one-hot 2D array for next move)
      - Labels the game phase (opening, midgame, endgame) based on move index thresholds
      - Assigns a value target based on the game outcome
      - If the original board size is not equal to the desired output_size,
        performs a center crop.
    
    Args:
        moves (list of tuples): List of moves as (x, y, time).
        board_size (int): Original board size from the file.
        winner (int): 1 if black wins, -1 if white wins, 0 if draw
        output_size (int): Desired board size for the network (e.g., 15).
        opening_threshold (int): Move index threshold for labeling opening phase.
        midgame_threshold (int): Move index threshold for labeling midgame phase.
        
    Returns:
        examples (list): List of tuples (board_state, target, phase, value_target) where:
            - board_state: np.ndarray of shape (output_size, output_size, 3)
            - target: np.ndarray of shape (output_size, output_size) (one-hot)
            - phase: string indicating game phase ('opening', 'midgame', 'endgame')
            - value_target: float indicating the expected outcome for the current player
    """
    # Initialize board: channels are (black, white, empty).
    # Start with an empty board: empty channel is 1 everywhere.
    start_time = time.time()
    board = np.zeros((board_size, board_size, 3), dtype=np.int32)
    board[:, :, 2] = 1  # channel 2 represents empty cells
    examples = []
    
    # Track sequence validity
    invalid_moves_count = 0
    max_invalid_moves = 5  # Threshold for considering a game corrupted
    
    # Track move sequence for player alternation checking
    player_sequence = []
    
    # Track examples per phase for debugging
    phase_counts = {'opening': 0, 'midgame': 0, 'endgame': 0}
    
    # Track skipped moves
    skipped_moves = 0
    
    debug_print(f"Generating examples from {len(moves)} moves (board_size={board_size}, output_size={output_size})", 2)
    if winner is not None:
        debug_print(f"Game outcome: {'Black wins' if winner == 1 else 'White wins' if winner == -1 else 'Draw'}", 3)
    else:
        debug_print("Game outcome: Undetermined", 3)
    
    # Process moves: we use alternating moves to assign players.
    # Assuming Black plays first (even-indexed moves) and White second.
    for i in range(len(moves) - 1):
        x, y, _ = moves[i]
        next_move = moves[i+1]
        # Convert coordinates from 1-indexed to 0-indexed
        x_idx = x - 1
        y_idx = y - 1
        
        # Validate current move coordinates
        if not (0 <= x_idx < board_size and 0 <= y_idx < board_size):
            invalid_moves_count += 1
            debug_print(f"Invalid move {i+1}: ({x},{y}) - Out of bounds", 3)
            continue
        
        # Validate move: skip if the position is already occupied
        if board[x_idx, y_idx, 2] != 1:
            invalid_moves_count += 1
            debug_print(f"Invalid move {i+1}: ({x},{y}) - Position already occupied", 3)
            continue
        
        # Determine current player (0 for black, 1 for white)
        current_player = i % 2  # 0 for black (first player), 1 for white (second player)
        
        # Track player sequence to detect out-of-order moves
        player_sequence.append(current_player)
        if len(player_sequence) >= 3:
            # Check if the same player moved twice in a row
            if player_sequence[-1] == player_sequence[-2]:
                invalid_moves_count += 1
                debug_print(f"Invalid player sequence at move {i+1}: {player_sequence[-2]}->{player_sequence[-1]}", 3)
                continue
        
        # Place the stone: channel 0 for Black, channel 1 for White
        if current_player == 0:  # Black's move
            board[x_idx, y_idx, 0] = 1
        else:
            board[x_idx, y_idx, 1] = 1
        board[x_idx, y_idx, 2] = 0  # mark as no longer empty
        
        # Validate next move coordinates
        next_x, next_y, _ = next_move
        next_x_idx = next_x - 1
        next_y_idx = next_y - 1
        
        if not (0 <= next_x_idx < board_size and 0 <= next_y_idx < board_size):
            invalid_moves_count += 1
            debug_print(f"Invalid next move {i+2}: ({next_x},{next_y}) - Out of bounds", 3)
            continue
        
        # If the game has too many invalid moves, it's likely corrupted - skip it entirely
        if invalid_moves_count > max_invalid_moves:
            debug_print(f"Too many invalid moves ({invalid_moves_count} > {max_invalid_moves}), discarding game", 2)
            return []  # Return empty list to indicate corrupted game
        
        # Prepare input: board state *after* current move (before the next move)
        board_state = board.copy()
        
        # Create the target: one-hot encoding for next move location.
        target = np.zeros((board_size, board_size), dtype=np.int32)
        
        # Validate that the next move is legal (the target cell should be empty)
        if board_state[next_x_idx, next_y_idx, 2] != 1:
            invalid_moves_count += 1
            debug_print(f"Invalid next move {i+2}: ({next_x},{next_y}) - Position already occupied", 3)
            continue
            
        target[next_x_idx, next_y_idx] = 1
        
        # If board size differs from the network's expected output_size, perform a center crop.
        if board_size != output_size:
            debug_print(f"Performing center crop from {board_size}x{board_size} to {output_size}x{output_size}", 3)
            board_state = center_crop(board_state, output_size)
            target = center_crop(target, output_size)
        
        # Label the game phase using move index thresholds.
        if i < opening_threshold:
            phase = 'opening'
        elif i < midgame_threshold:
            phase = 'midgame'
        else:
            phase = 'endgame'
            
        phase_counts[phase] += 1
            
        # Determine value target: 1 if current player will win, -1 if will lose, 0 for draw
        # If winner == 1 (black), current_player == 0 wins
        # If winner == -1 (white), current_player == 1 wins
        if winner is None:
            value_target = 0.0  # Default if outcome is unknown
        elif winner == 0:
            value_target = 0.0  # Draw
        else:
            # Convert winner from (1/-1) to player index (0/1)
            winner_idx = 0 if winner == 1 else 1
            # If current player is the winner, value_target = 1, else -1
            value_target = 1.0 if current_player == winner_idx else -1.0
            
        examples.append((board_state, target, phase, value_target))
    
    # Debug statistics
    elapsed = time.time() - start_time
    debug_print(f"Generated {len(examples)} examples in {elapsed:.3f}s", 2)
    debug_print(f"Phase distribution - Opening: {phase_counts['opening']}, Midgame: {phase_counts['midgame']}, Endgame: {phase_counts['endgame']}", 3)
    debug_print(f"Invalid moves skipped: {invalid_moves_count}", 3)
    
    if invalid_moves_count > 0:
        debug_print(f"Warning: {invalid_moves_count} invalid moves detected in sequence", 2)
    
    return examples

def augment_data(board, target, value_target):
    """
    Augment a board state and its corresponding target by applying rotations and flips.
    
    The function generates 8 variations:
      - 0°, 90°, 180°, 270° rotations
      - Each rotation, followed by a horizontal flip
    
    Args:
        board (np.ndarray): Input board state of shape (H, W, 3).
        target (np.ndarray): Target one-hot board of shape (H, W).
        value_target (float): The value target (win probability).
        
    Returns:
        aug_examples (list): List of tuples (aug_board, aug_target, aug_value) for each augmentation.
    """
    aug_examples = []
    # Generate rotations: 0, 90, 180, 270 degrees.
    for k in range(4):
        board_rot = np.rot90(board, k=k)
        target_rot = np.rot90(target, k=k)
        aug_examples.append((board_rot, target_rot, value_target))
        # Apply horizontal flip after rotation.
        board_flip = np.fliplr(board_rot)
        target_flip = np.fliplr(target_rot)
        aug_examples.append((board_flip, target_flip, value_target))
    return aug_examples

def process_files(file_list, output_size=15, opening_threshold=10, midgame_threshold=20, 
                 batch_size=1000, save_batches=True, output_path="gomoku_dataset.npz"):
    """
    Process multiple .psq files to generate a complete dataset.
    
    For each file, the function:
      - Parses the file to extract board size, moves, and winner.
      - Generates training examples from the moves.
      - Applies data augmentation to each example.
      - Periodically saves batches to disk to reduce memory usage.
      
    Args:
        file_list (list of str): List of file paths to .psq files.
        output_size (int): Desired board size for training (e.g., 15).
        opening_threshold (int): Move index threshold for the opening phase.
        midgame_threshold (int): Move index threshold for the midgame phase.
        batch_size (int): Number of games to process before saving a batch.
        save_batches (bool): Whether to save intermediate batches.
        output_path (str): Base path for saving the dataset.
        
    Returns:
        If save_batches is True: List of paths to temporary batch files.
        Otherwise: Tuple of (boards, targets, phases, values) arrays.
    """
    all_boards = []
    all_targets = []
    all_phases = []
    all_values = []
    
    # Keep track of total examples across all batches
    total_examples_across_batches = 0
    batch_number = 0
    batch_paths = []
    
    # Statistics tracking
    file_count = len(file_list)
    processed_files = 0
    skipped_files = 0
    corrupted_games = 0
    valid_games = 0
    total_examples = 0
    error_files = []
    
    # Phase statistics
    opening_examples = 0
    midgame_examples = 0
    endgame_examples = 0
    
    # Value statistics
    win_examples = 0
    loss_examples = 0
    draw_examples = 0
    
    # Timing statistics
    start_time = time.time()
    last_progress_time = start_time
    parse_time = 0
    generate_time = 0
    augment_time = 0
    
    # Memory statistics
    initial_memory = get_memory_usage() if MONITOR_MEMORY else 0
    peak_memory = initial_memory
    
    debug_print(f"Starting to process {file_count} PSQ files...", 0)
    debug_print(f"Output board size: {output_size}x{output_size}", 1)
    debug_print(f"Phase thresholds - Opening: <{opening_threshold} moves, Midgame: <{midgame_threshold} moves", 1)
    
    if save_batches:
        debug_print(f"Using batch processing to reduce memory usage (temporary files will be created and merged at the end)", 1)
        debug_print(f"Batch size: {batch_size} games", 1)
    else:
        debug_print(f"Batch processing disabled - keeping all data in memory", 1)
    
    debug_print(f"Press Ctrl+C at any time to safely stop processing", 1)
    
    if MONITOR_MEMORY:
        debug_print(f"Initial memory usage: {initial_memory:.1f} MB", 1)
    
    progress_interval = max(1, min(100, file_count // 20))  # Report 20 times during processing or every file if small
    
    # Initialize progress bar
    if USE_PROGRESS_BAR:
        print_progress_bar(0, file_count, prefix='Progress:', suffix='Complete', length=50)
    
    for file_idx, file in enumerate(file_list):
        # Check for interrupt request
        if INTERRUPT_REQUESTED:
            debug_print(f"Processing stopped after {file_idx} files due to user interrupt", 0)
            break
            
        file_start_time = time.time()
        try:
            # Update progress bar
            if USE_PROGRESS_BAR:
                suffix = f'Complete ({file_idx+1}/{file_count})'
                if MONITOR_MEMORY:
                    current_memory = get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    suffix += f' - Mem: {current_memory:.1f} MB'
                print_progress_bar(file_idx + 1, file_count, prefix='Progress:', 
                                  suffix=suffix, length=50)
            
            # Periodically report progress if not using progress bar
            current_time = time.time()
            elapsed = current_time - start_time
            if not USE_PROGRESS_BAR and file_idx % progress_interval == 0:
                if file_idx > 0:
                    progress_pct = file_idx / file_count * 100
                    elapsed_since_last = current_time - last_progress_time
                    files_since_last = progress_interval
                    files_per_second = files_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
                    estimated_remaining = (file_count - file_idx) / files_per_second if files_per_second > 0 else 0
                    
                    debug_print(f"Progress: {file_idx}/{file_count} files ({progress_pct:.1f}%) - " +
                           f"Speed: {files_per_second:.1f} files/s - " +
                           f"Est. remaining: {estimated_remaining/60:.1f} min", 0)
                    debug_print(f"Examples in current batch: {len(all_boards)} (Valid games: {valid_games}, " +
                           f"Total examples so far: {total_examples_across_batches + len(all_boards)})", 1)
                    
                    if MONITOR_MEMORY:
                        current_memory = get_memory_usage()
                        peak_memory = max(peak_memory, current_memory)
                        memory_increase = current_memory - initial_memory
                        debug_print(f"Memory usage: {current_memory:.1f} MB (+{memory_increase:.1f} MB, Peak: {peak_memory:.1f} MB)", 1)
                    
                    last_progress_time = current_time
            
            debug_print(f"Processing file {file_idx+1}/{file_count}: {os.path.basename(file)}", 2)
            
            # Time the parsing
            parse_start = time.time()
            board_size, moves, winner = parse_psq_file(file)
            parse_time += time.time() - parse_start
            
            # Skip if parsing failed or no moves
            if board_size is None or not moves:
                skipped_files += 1
                error_files.append(file)
                debug_print(f"Skipping file due to parsing failure: {file}", 2)
                continue
                
            debug_print(f"File parsed successfully: {len(moves)} moves, board size {board_size}x{board_size}", 3)
            
            # Time the example generation
            gen_start = time.time()
            examples = generate_training_examples(moves, board_size, winner, output_size,
                                                opening_threshold, midgame_threshold)
            generate_time += time.time() - gen_start
            
            # Check if game is corrupted (empty examples list)
            if not examples:
                corrupted_games += 1
                error_files.append(file)
                debug_print(f"Game detected as corrupted, skipping: {file}", 2)
                continue
                
            debug_print(f"Generated {len(examples)} training examples from file", 3)
                
            valid_games += 1
            processed_files += 1
            
            # Time the augmentation
            aug_start = time.time()
            for board, target, phase, value_target in examples:
                # Track phase statistics
                if phase == 'opening':
                    opening_examples += 1
                elif phase == 'midgame':
                    midgame_examples += 1
                else:  # endgame
                    endgame_examples += 1
                
                # Track value statistics
                if value_target > 0:
                    win_examples += 1
                elif value_target < 0:
                    loss_examples += 1
                else:
                    draw_examples += 1
                
                # Augment each example and add the augmented versions.
                aug_examples = augment_data(board, target, value_target)
                for aug_board, aug_target, aug_value in aug_examples:
                    all_boards.append(aug_board)
                    all_targets.append(aug_target)
                    all_phases.append(phase)
                    all_values.append(aug_value)
                    total_examples += 1
            augment_time += time.time() - aug_start
            
            # Save batch if we've processed enough games
            if save_batches and valid_games % batch_size == 0 and all_boards:
                batch_number += 1
                batch_path = f"{os.path.splitext(output_path)[0]}_batch{batch_number}_temp.npz"
                debug_print(f"Saving temporary batch {batch_number} with {len(all_boards)} examples", 1)
                
                # Convert lists to numpy arrays
                boards_array = np.array(all_boards, dtype=np.int8)
                targets_array = np.array(all_targets, dtype=np.int8)
                phases_array = np.array(all_phases)
                values_array = np.array(all_values, dtype=np.float32)
                
                # Save batch
                np.savez_compressed(batch_path, 
                                   boards=boards_array, 
                                   targets=targets_array, 
                                   phases=phases_array, 
                                   values=values_array)
                
                # Record batch path for later merging
                batch_paths.append(batch_path)
                
                # Update total examples count and clear lists to free memory
                total_examples_across_batches += len(all_boards)
                all_boards = []
                all_targets = []
                all_phases = []
                all_values = []
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                if MONITOR_MEMORY:
                    current_memory = get_memory_usage()
                    debug_print(f"Memory after batch save: {current_memory:.1f} MB", 1)
                
        except Exception as e:
            debug_print(f"Error processing file {file}: {str(e)}", 1)
            import traceback
            traceback.print_exc()
            skipped_files += 1
            error_files.append(file)
        
        file_time = time.time() - file_start_time
        debug_print(f"File processing time: {file_time:.3f}s", 3)
        
        # Check memory usage after each file if monitoring is enabled
        if MONITOR_MEMORY and file_idx % 10 == 0:
            current_memory = get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
            debug_print(f"Memory usage after {file_idx+1} files: {current_memory:.1f} MB", 3)
    
    # Ensure progress bar completes
    if USE_PROGRESS_BAR:
        if INTERRUPT_REQUESTED:
            suffix = f'Stopped at {processed_files}/{file_count}'
        else:
            suffix = 'Complete'
            
        if MONITOR_MEMORY:
            current_memory = get_memory_usage()
            suffix += f' - Mem: {current_memory:.1f} MB'
        print_progress_bar(processed_files, file_count, prefix='Progress:', suffix=suffix, length=50)
    
    total_elapsed = time.time() - start_time
    
    # Check if processing was interrupted
    if INTERRUPT_REQUESTED:
        debug_print(f"\nProcessing was interrupted by user after {processed_files}/{file_count} files", 0)
    
    # Memory statistics at the end
    if MONITOR_MEMORY:
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        debug_print(f"\nMemory Statistics:", 0)
        debug_print(f"- Initial memory: {initial_memory:.1f} MB", 0)
        debug_print(f"- Final memory: {final_memory:.1f} MB", 0)
        debug_print(f"- Memory increase: {memory_increase:.1f} MB", 0)
        debug_print(f"- Peak memory: {peak_memory:.1f} MB", 0)
    
    # Save final batch if there are any examples remaining
    if all_boards:
        if save_batches:
            batch_number += 1
            batch_path = f"{os.path.splitext(output_path)[0]}_batch{batch_number}_temp.npz"
            debug_print(f"Saving final temporary batch {batch_number} with {len(all_boards)} examples", 1)
            
            # Convert lists to numpy arrays
            boards_array = np.array(all_boards, dtype=np.int8)
            targets_array = np.array(all_targets, dtype=np.int8)
            phases_array = np.array(all_phases)
            values_array = np.array(all_values, dtype=np.float32)
            
            # Save batch
            np.savez_compressed(batch_path, 
                               boards=boards_array, 
                               targets=targets_array, 
                               phases=phases_array, 
                               values=values_array)
            
            # Record batch path for later merging
            batch_paths.append(batch_path)
            
            # Update total examples count
            total_examples_across_batches += len(all_boards)
            final_example_count = total_examples_across_batches
        else:
            # If we're not saving batches, count the current examples
            final_example_count = len(all_boards)
    else:
        # All examples were already saved in batches
        final_example_count = total_examples_across_batches
    
    # Detailed processing statistics
    debug_print("\nProcessing Statistics:", 0)
    debug_print(f"- Total processing time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)", 0)
    debug_print(f"- Parse time: {parse_time:.2f}s ({parse_time/total_elapsed*100:.1f}%)", 1)
    debug_print(f"- Example generation time: {generate_time:.2f}s ({generate_time/total_elapsed*100:.1f}%)", 1)
    debug_print(f"- Augmentation time: {augment_time:.2f}s ({augment_time/total_elapsed*100:.1f}%)", 1)
    debug_print(f"- Files processed successfully: {processed_files}/{file_count} ({processed_files/file_count*100:.1f}%)", 0)
    debug_print(f"- Files skipped/errored: {skipped_files}/{file_count} ({skipped_files/file_count*100:.1f}%)", 0)
    debug_print(f"- Valid games extracted: {valid_games}", 0)
    debug_print(f"- Corrupted games detected and skipped: {corrupted_games}", 0)
    debug_print(f"- Total training examples generated: {final_example_count}", 0)
    
    # Phase and value distribution (only if we have examples)
    original_examples = opening_examples + midgame_examples + endgame_examples
    if original_examples > 0:
        debug_print("\nExample Distribution by Phase:", 1)
        debug_print(f"- Opening: {opening_examples} ({opening_examples/original_examples*100:.1f}%)", 1)
        debug_print(f"- Midgame: {midgame_examples} ({midgame_examples/original_examples*100:.1f}%)", 1)
        debug_print(f"- Endgame: {endgame_examples} ({endgame_examples/original_examples*100:.1f}%)", 1)
        
        debug_print("\nExample Distribution by Outcome:", 1)
        debug_print(f"- Win: {win_examples} ({win_examples/original_examples*100:.1f}%)", 1)
        debug_print(f"- Loss: {loss_examples} ({loss_examples/original_examples*100:.1f}%)", 1)
        debug_print(f"- Draw: {draw_examples} ({draw_examples/original_examples*100:.1f}%)", 1)
    
    # Log files with errors if any
    if error_files:
        log_filename = "psq_processing_errors.log"
        with open(log_filename, "w") as f:
            f.write(f"Files with processing errors or skipped ({len(error_files)}):\n")
            for file in error_files:
                f.write(f"{file}\n")
        debug_print(f"- Error files logged to {log_filename}", 0)
    
    # Safety check for empty dataset
    if final_example_count == 0:
        error_msg = "No valid examples generated. Check PSQ files and processing errors."
        debug_print(f"ERROR: {error_msg}", 0)
        raise ValueError(error_msg)
    
    # Return either the full dataset or info about batches
    if save_batches:
        if batch_paths:
            debug_print(f"Created {len(batch_paths)} temporary batch files that will be merged into the final dataset", 1)
        return batch_paths
    else:
        # Convert lists to numpy arrays (may still fail if too large)
        debug_print(f"Converting data to numpy arrays (this may require significant memory)...", 1)
        try:
            # Use more memory-efficient data types
            boards_array = np.array(all_boards, dtype=np.int8)
            targets_array = np.array(all_targets, dtype=np.int8)
            phases_array = np.array(all_phases)
            values_array = np.array(all_values, dtype=np.float32)
            
            debug_print(f"Conversion successful", 1)
            return boards_array, targets_array, phases_array, values_array
        except MemoryError as e:
            debug_print(f"Memory error during array conversion: {str(e)}", 0)
            debug_print(f"Suggestion: Re-run with batch processing enabled (save_batches=True)", 0)
            raise

def merge_batches(batch_paths, output_path="gomoku_dataset.npz", delete_batches=True):
    """
    Merge multiple batch files into a single dataset.
    
    Args:
        batch_paths (list): List of paths to batch files.
        output_path (str): Path for the merged dataset.
        delete_batches (bool): Whether to delete batch files after successful merge.
        
    Returns:
        Path to the merged dataset.
    """
    debug_print(f"Merging {len(batch_paths)} batch files into a single dataset: {output_path}...", 0)
    
    # Check if the batch paths exist
    for path in batch_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Batch file not found: {path}")
    
    # Collect data from all batches
    all_boards = []
    all_targets = []
    all_phases = []
    all_values = []
    total_examples = 0
    
    start_time = time.time()
    
    # Process each batch
    for i, batch_path in enumerate(batch_paths):
        debug_print(f"Loading batch {i+1}/{len(batch_paths)}: {os.path.basename(batch_path)}", 1)
        batch_start = time.time()
        
        try:
            # Load batch
            batch_data = np.load(batch_path)
            
            # Get data and count examples
            boards = batch_data["boards"]
            targets = batch_data["targets"]
            phases = batch_data["phases"]
            values = batch_data["values"]
            
            batch_examples = len(boards)
            total_examples += batch_examples
            
            # Append to lists (memory inefficient but keeps the code simple)
            all_boards.append(boards)
            all_targets.append(targets)
            all_phases.append(phases)
            all_values.append(values)
            
            batch_time = time.time() - batch_start
            debug_print(f"Loaded {batch_examples} examples in {batch_time:.2f}s", 2)
            
        except Exception as e:
            debug_print(f"Error loading batch {batch_path}: {str(e)}", 0)
            raise
    
    # Concatenate all data (this is the memory-intensive part)
    debug_print(f"Concatenating {total_examples} examples from all batches...", 1)
    concat_start = time.time()
    
    try:
        # Concatenate along the first axis (examples)
        merged_boards = np.concatenate(all_boards, axis=0)
        merged_targets = np.concatenate(all_targets, axis=0)
        merged_phases = np.concatenate(all_phases, axis=0)
        merged_values = np.concatenate(all_values, axis=0)
        
        concat_time = time.time() - concat_start
        debug_print(f"Concatenation completed in {concat_time:.2f}s", 1)
        
        # Save the merged dataset
        debug_print(f"Saving final dataset with {len(merged_boards)} examples to {output_path}", 0)
        save_start = time.time()
        
        np.savez_compressed(output_path, 
                           boards=merged_boards,
                           targets=merged_targets,
                           phases=merged_phases,
                           values=merged_values)
        
        save_time = time.time() - save_start
        debug_print(f"Final dataset saved in {save_time:.2f}s", 1)
        
        # Delete batch files if requested
        if delete_batches:
            debug_print(f"Cleaning up intermediate batch files...", 1)
            for batch_path in batch_paths:
                try:
                    os.remove(batch_path)
                    debug_print(f"Deleted: {batch_path}", 3)
                except Exception as e:
                    debug_print(f"Warning: Failed to delete batch file {batch_path}: {str(e)}", 1)
        
        total_time = time.time() - start_time
        debug_print(f"Total merging time: {total_time:.2f}s", 0)
        debug_print(f"All data successfully combined into a single file: {output_path}", 0)
        
        return output_path
        
    except MemoryError as e:
        debug_print(f"Memory error during merge: {str(e)}", 0)
        debug_print("The dataset is too large to merge in memory.", 0)
        debug_print("You can still use the individual batch files for training, or try on a machine with more memory.", 0)
        raise

def save_dataset(npz_path, boards, targets, phases, values):
    """
    Save the processed dataset to a compressed .npz file.
    
    Args:
        npz_path (str): Output file path for the dataset.
        boards (np.ndarray): Input boards array.
        targets (np.ndarray): Target moves array.
        phases (np.ndarray): Array of game phase labels.
        values (np.ndarray): Array of value targets.
    """
    start_time = time.time()
    debug_print(f"Saving dataset to {npz_path}...", 0)
    
    # If boards is a list of batch paths, merge them
    if isinstance(boards, list) and all(isinstance(path, str) for path in boards):
        debug_print(f"Detected batch paths, merging batches...", 1)
        merge_batches(boards, npz_path)
        return
    
    # Calculate and display dataset statistics
    file_size_mb = (boards.nbytes + targets.nbytes + phases.nbytes + values.nbytes) / (1024 * 1024)
    debug_print(f"Dataset info:", 1)
    debug_print(f"- Examples: {len(boards)}", 1)
    debug_print(f"- Board shape: {boards.shape}", 1)
    debug_print(f"- Target shape: {targets.shape}", 1)
    debug_print(f"- Estimated file size: {file_size_mb:.1f} MB", 1)
    
    # Save with compression
    np.savez_compressed(npz_path, boards=boards, targets=targets, phases=phases, values=values)
    
    # Get actual file size
    actual_file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    compression_ratio = file_size_mb / actual_file_size_mb if actual_file_size_mb > 0 else 0
    
    elapsed = time.time() - start_time
    debug_print(f"Dataset saved successfully in {elapsed:.2f}s", 0)
    debug_print(f"- Actual file size: {actual_file_size_mb:.1f} MB", 1)
    debug_print(f"- Compression ratio: {compression_ratio:.1f}x", 1)
    debug_print(f"- Average example size: {(actual_file_size_mb * 1024 * 1024 / len(boards)):.1f} bytes", 2)

# Example main function that processes all .psq files in a directory.
if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments (still available if needed)
    parser = argparse.ArgumentParser(description="Process PSQ game files to generate a training dataset for the Gomoku engine.")
    parser.add_argument("--psq_dir", type=str, default="psq_games", help="Directory containing PSQ files")
    parser.add_argument("--output", type=str, default="gomoku_dataset.npz", help="Output NPZ file path")
    parser.add_argument("--board_size", type=int, default=15, help="Output board size (default: 15)")
    parser.add_argument("--opening_threshold", type=int, default=10, help="Threshold for opening phase (default: 10 moves)")
    parser.add_argument("--midgame_threshold", type=int, default=20, help="Threshold for midgame phase (default: 20 moves)")
    parser.add_argument("--debug_level", type=int, default=2, choices=[0, 1, 2, 3], 
                        help="Debug verbosity level: 0=minimal, 1=normal, 2=detailed, 3=very detailed")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (default: all)")
    parser.add_argument("--progress_bar", action="store_true", help="Display a progress bar instead of textual progress updates")
    parser.add_argument("--monitor_memory", action="store_true", help="Monitor and report memory usage during processing")
    parser.add_argument("--no-enhanced-debug", action="store_true", help="Disable enhanced debugging (use CLI args only)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of games to process before saving a temporary batch")
    parser.add_argument("--no-batches", action="store_true", help="Disable batch processing (keep all data in memory)")
    parser.add_argument("--keep_temp_files", action="store_true", help="Keep temporary batch files after merging")
    parser.add_argument("--merge_only", action="store_true", help="Only merge existing batches without processing files")
    parser.add_argument("--batch_pattern", type=str, help="Pattern for batch files to merge (e.g., 'gomoku_dataset_batch*.npz')")
    
    args = parser.parse_args()
    
    # Automatically enable enhanced debugging features unless explicitly disabled
    if not args.no_enhanced_debug:
        set_debug_level(2)  # Detailed debugging by default
        set_use_progress_bar(True)  # Enable progress bar by default
        set_monitor_memory(True)  # Will be disabled automatically if psutil not available
    else:
        # Use values from command line args
        set_debug_level(args.debug_level)
        set_use_progress_bar(args.progress_bar)
        set_monitor_memory(args.monitor_memory)
    
    debug_print(f"Starting datapipeline with enhanced debugging (debug level {DEBUG_LEVEL})", 0)
    debug_print(f"Progress bar: {'Enabled' if USE_PROGRESS_BAR else 'Disabled'}", 1)
    debug_print(f"Memory monitoring: {'Enabled' if MONITOR_MEMORY else 'Disabled'}", 1)
    
    # If merge_only mode is enabled, just merge existing batches
    if args.merge_only:
        if not args.batch_pattern:
            debug_print("Error: --batch_pattern is required with --merge_only", 0)
            sys.exit(1)
            
        debug_print(f"Merge-only mode: Finding batches matching '{args.batch_pattern}'", 0)
        batch_paths = glob.glob(args.batch_pattern)
        
        if not batch_paths:
            debug_print(f"No batch files found matching pattern: {args.batch_pattern}", 0)
            sys.exit(1)
            
        debug_print(f"Found {len(batch_paths)} batch files to merge", 0)
        
        try:
            output_path = args.output
            merge_batches(batch_paths, output_path, delete_batches=not args.keep_temp_files)
            debug_print(f"Merge completed successfully. Final dataset saved to {output_path}", 0)
            sys.exit(0)
        except Exception as e:
            debug_print(f"Error during merge: {str(e)}", 0)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Regular processing mode
    debug_print(f"Parameters:", 1)
    debug_print(f"- Input directory: {args.psq_dir}", 1)
    debug_print(f"- Output file: {args.output}", 1)
    debug_print(f"- Board size: {args.board_size}x{args.board_size}", 1)
    debug_print(f"- Opening threshold: {args.opening_threshold}", 1)
    debug_print(f"- Midgame threshold: {args.midgame_threshold}", 1)
    debug_print(f"- Batch processing: {'Disabled' if args.no_batches else 'Enabled'}", 1)
    if not args.no_batches:
        debug_print(f"- Batch size: {args.batch_size} games", 1)
        debug_print(f"- Temporary files: {'Kept' if args.keep_temp_files else 'Auto-deleted after merging'}", 1)
    if args.limit:
        debug_print(f"- File limit: {args.limit}", 1)
    
    # Find all PSQ files in all subdirectories
    psq_files = []
    for root, dirs, files in os.walk(args.psq_dir):
        for file in files:
            if file.endswith(".psq"):
                psq_files.append(os.path.join(root, file))
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        original_count = len(psq_files)
        psq_files = psq_files[:args.limit]
        debug_print(f"Limited processing to {len(psq_files)}/{original_count} files", 0)
    
    debug_print(f"Found {len(psq_files)} PSQ files", 0)
    
    # Process files to generate dataset
    try:
        start_time = time.time()
        result = process_files(
            psq_files, 
            output_size=args.board_size,
            opening_threshold=args.opening_threshold, 
            midgame_threshold=args.midgame_threshold,
            batch_size=args.batch_size,
            save_batches=not args.no_batches,
            output_path=args.output
        )
        elapsed = time.time() - start_time
        
        # The result could be batch paths or actual arrays depending on save_batches
        if isinstance(result, list) and all(isinstance(path, str) for path in result):
            # Batch paths returned
            batch_count = len(result)
            debug_print(f"Processing completed in {elapsed:.2f}s ({elapsed/60:.2f} min)", 0)
            
            # Merge batches into the final dataset
            if batch_count > 0:
                debug_print(f"Merging temporary batches into final dataset: {args.output}", 0)
                merge_batches(result, args.output, delete_batches=not args.keep_temp_files)
            else:
                debug_print(f"No batches were created (no valid examples found)", 0)
        else:
            # Arrays returned (no batches mode)
            boards, targets, phases, values = result
            debug_print(f"Processing completed in {elapsed:.2f}s ({elapsed/60:.2f} min)", 0)
            debug_print(f"Total examples: {len(boards)}", 0)
    
    # Save the dataset
            save_dataset(args.output, boards, targets, phases, values)
            
    except Exception as e:
        debug_print(f"Error during processing: {str(e)}", 0)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    debug_print(f"All processing completed successfully - final dataset saved to: {args.output}", 0)

# Add these deprecated function stubs at the end of the file

def has_significant_threats(board, player):
    """
    Deprecated: This function is no longer used with the exact win checking approach.
    Kept as a stub for compatibility.
    """
    return False

def game_nearly_over(board, player):
    """
    Deprecated: This function is no longer used with the exact win checking approach.
    Kept as a stub for compatibility.
    """
    return False

def check_overline(board, x, y, player):
    """
    Deprecated: This function is no longer used with the exact win checking approach.
    Kept as a stub for compatibility.
    """
    return False

def is_winning_move(board, x, y, player):
    """
    Deprecated: This function is now replaced by check_win.
    Kept as a stub for compatibility.
    """
    return check_win(board, x, y, player)

def count_open_threes(board, player):
    """
    Deprecated: This function is no longer used with the exact win checking approach.
    Kept as a stub for compatibility.
    """
    return 0

def count_open_fours(board, player):
    """
    Deprecated: This function is no longer used with the exact win checking approach.
    Kept as a stub for compatibility.
    """
    return 0

def is_player_blocked(board, player):
    """
    Deprecated: This function is no longer used with the exact win checking approach.
    Kept as a stub for compatibility.
    """
    return False
