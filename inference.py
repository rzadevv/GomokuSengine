import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GomokuNet  # Ensure model.py is in the same directory or in your PYTHONPATH
import gc  # Add garbage collection import

def load_model(model_path, device='cuda'):
    """
    Load the trained GomokuNet model from the checkpoint.
    
    Args:
        model_path (str): Path to the saved model checkpoint.
        device (str): Device to load the model onto ('cuda' or 'cpu').
    
    Returns:
        model: The loaded GomokuNet model set to evaluation mode.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = GomokuNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Run garbage collection after model loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model

def raw_board_to_tensor(raw_board):
    """
    Convert a raw board (numpy array) to a torch tensor.
    
    The expected format for the raw board is (15,15,3):
      - Channel 0: Black stones
      - Channel 1: White stones
      - Channel 2: Empty cells
      
    Returns:
        Tensor of shape (1, 3, 15, 15)
    """
    # Convert to float tensor and permute dimensions to (channels, height, width)
    board_tensor = torch.tensor(raw_board, dtype=torch.float32)
    board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0)  # add batch dimension
    return board_tensor

def predict_move(model, raw_board, temperature=1.0, move_count=0, device='cuda', top_k=5, use_value_eval=True):
    """
    Predict the best move for a given raw board.
    
    Uses both policy and value heads for decision making. First gets top-K policy moves,
    then evaluates each by simulating the position after the move and selecting the one 
    with the best expected outcome.
    
    Args:
        model: Loaded GomokuNet model.
        raw_board (np.array): Board in shape (15,15,3).
        temperature (float): Temperature scaling factor.
        move_count (int): Current move count in the game, used for early game exploration.
        device (str): 'cuda' or 'cpu'.
        top_k (int): Number of top policy moves to evaluate with the value head.
        use_value_eval (bool): Whether to use value head to evaluate candidate moves.
                               If False, falls back to policy-only decision.
        
    Returns:
        top_move (tuple): (row, col) for the selected move (1-indexed).
        probabilities (np.array): Array of probabilities of shape (15,15).
        value (float): The model's evaluation of the position (-1 to 1).
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Ensure consistent precision (float32)
    board_tensor = raw_board_to_tensor(raw_board).to(device, dtype=torch.float32)
    
    # Initial forward pass to get policy and value for current position
    with torch.no_grad():
        try:
            # Handle both regular and TorchScript models
            if isinstance(model, torch.jit.ScriptModule):
                # TorchScript model
                output = model(board_tensor)
                policy_logits, value_pred = output
            else:
                # Regular model
                policy_logits, value_pred = model(board_tensor)  # shapes: (1, 225), (1, 1)
                
            # Apply temperature scaling: divide logits by temperature
            scaled_logits = policy_logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)  # shape: (1, 225)
            initial_value = value_pred.item()  # Get the scalar value for initial position
            
            # Free up memory after the forward pass
            if 'output' in locals():
                del output
        except Exception as e:
            print(f"Error during model inference: {e}")
            # Fallback to random valid move
            empty_mask = raw_board[:, :, 2] == 1.0
            valid_positions = np.where(empty_mask)
            if len(valid_positions[0]) > 0:
                # Choose random empty position
                idx = np.random.randint(len(valid_positions[0]))
                row, col = valid_positions[0][idx], valid_positions[1][idx]
                # Return 1-indexed position
                return (row + 1, col + 1), np.zeros((15, 15)), 0.0
            else:
                # No valid moves
                return None, np.zeros((15, 15)), 0.0
    
    # Convert probabilities to a 15x15 numpy array
    probs_np = probs.cpu().numpy().reshape(15, 15)
    
    # Make sure we don't place a stone on an already occupied position
    mask = raw_board[:, :, 2] != 1.0
    probs_np[mask] = 0.0
    
    # Free memory from tensors we no longer need
    del board_tensor, policy_logits, scaled_logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force randomness for all moves when temperature is higher
    if temperature > 0.5 or move_count < 3:
        # Flatten probabilities
        flat_probs = probs_np.flatten()
        
        # Ensure the probabilities sum to 1
        if np.sum(flat_probs) > 0:
            flat_probs = flat_probs / np.sum(flat_probs)
            
            # Sample a move based on the probabilities
            # If there are any valid moves, sample one
            legal_indices = np.where(flat_probs > 0)[0]
            if len(legal_indices) > 0:
                # Sample from all legal moves proportionally to their probabilities
                sampled_idx = np.random.choice(len(flat_probs), p=flat_probs)
                
                # Convert to 2D coordinates
                row, col = sampled_idx // 15, sampled_idx % 15
                
                # Convert to 1-indexed coordinates
                top_move = (row + 1, col + 1)
                return top_move, probs_np, initial_value
    
    # For early moves (moves 0, 1, 2), use Top-K sampling for exploration
    if move_count < 3 and not use_value_eval:  # Only use this when not doing value evaluation
        # Determine K based on move count (more diversity early)
        k_values = [10, 7, 5]  # K for moves 0, 1, 2
        k = k_values[move_count]
        
        # Flatten probabilities
        flat_probs = probs_np.flatten()
        
        # Find indices of the top K probabilities
        top_k_indices = np.argsort(flat_probs)[-k:]
        
        # Get the corresponding probabilities
        top_k_probs = flat_probs[top_k_indices]
        
        # Normalize to sum to 1
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # Sample one index based on the probabilities
        sampled_idx = np.random.choice(top_k_indices, p=top_k_probs)
        
        # Convert to 2D coordinates
        row, col = sampled_idx // 15, sampled_idx % 15
        
        # Convert to 1-indexed coordinates
        top_move = (row + 1, col + 1)
        return top_move, probs_np, initial_value
    
    # If we're using value evaluation (Method 2) or past the early game exploration phase:
    if use_value_eval:
        # Get the top K legal moves from policy probabilities
        flat_probs = probs_np.flatten()
        # Get indices of all non-zero probabilities (legal moves)
        legal_indices = np.where(flat_probs > 0)[0]
        
        # If no legal moves found, return an error
        if len(legal_indices) == 0:
            raise ValueError("No legal moves available")
        
        # Sort by probability (ascending) and take the top K
        sorted_indices = legal_indices[np.argsort(flat_probs[legal_indices])]
        # Get the K highest probability moves
        top_indices = sorted_indices[-min(top_k, len(sorted_indices)):][::-1]  # Reverse to start with highest prob
        
        # If temperature is > 0.1, add randomness by sampling from the top moves
        if temperature > 0.1 and len(top_indices) > 1:
            # Get probabilities for these top moves
            top_probs = flat_probs[top_indices]
            
            # Normalize to sum to 1
            top_probs = top_probs / np.sum(top_probs)
            
            # Sample one of the top indices based on their relative probabilities
            sampled_top_idx = np.random.choice(len(top_indices), p=top_probs)
            sampled_idx = top_indices[sampled_top_idx]
            
            # Convert to 2D coordinates
            row, col = sampled_idx // 15, sampled_idx % 15
            
            # Convert to 1-indexed coordinates
            top_move = (row + 1, col + 1)
            return top_move, probs_np, initial_value
        
        # Determine if we're black (True) or white (False)
        # Count stones to determine whose turn it is
        black_stones = np.sum(raw_board[:, :, 0])
        white_stones = np.sum(raw_board[:, :, 1])
        current_player = black_stones <= white_stones  # True if black's turn, False if white's turn
        
        # Initialize variables to track the best move
        best_move = None
        best_value_for_us = -float('inf')  # We want to maximize our value
        best_initial_prob = -1
        
        # Evaluate each of the top K moves
        for idx in top_indices:
            # Convert flat index to 2D coordinates (0-indexed)
            row, col = idx // 15, idx % 15
            # Convert to 1-indexed coordinates for place_stone
            candidate_move = (row + 1, col + 1)
            
            # Simulate making this move
            try:
                next_board = place_stone(raw_board, candidate_move, current_player=current_player)
                
                # Prepare the board tensor for evaluation
                next_board_tensor = raw_board_to_tensor(next_board).to(device)
                
                # Forward pass to get value prediction for the resulting position
                with torch.no_grad():
                    _, next_value_pred = model(next_board_tensor)
                    next_value = next_value_pred.item()
                
                # For the opponent's perspective, invert the value
                # (negative because opponent's win is our loss)
                opponent_expected_value = next_value
                our_expected_value = -opponent_expected_value  # We want opponent to have a bad position
                
                # If this move gives us a better expected outcome than previous best
                if our_expected_value > best_value_for_us:
                    best_value_for_us = our_expected_value
                    best_move = candidate_move
                    best_initial_prob = probs_np[row, col]
                    
                # Clean up memory for this evaluation
                del next_board_tensor, next_value_pred
                
            except ValueError as e:
                # Skip invalid moves - though this shouldn't happen if masking worked correctly
                continue
        
        # Run garbage collection after all evaluations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        # If we found a valid move through value evaluation
        if best_move is not None:
            # Return the best move, the original policy probabilities, and initial position value
            return best_move, probs_np, initial_value
            
    # Fallback to using highest probability move if value evaluation failed
    row, col = np.unravel_index(np.argmax(probs_np), probs_np.shape)
    top_move = (row + 1, col + 1)
    return top_move, probs_np, initial_value

def predict_move_with_details(model, raw_board, temperature=1.0, move_count=0, device='cuda', top_k=5, use_value_eval=True):
    """
    Extended version of predict_move that returns detailed information about evaluated moves.
    
    This function is identical to predict_move, but returns additional information about
    the evaluation process, which can be used for visualization and debugging.
    
    Returns:
        top_move (tuple): (row, col) for the selected move (1-indexed).
        probs_np (np.array): Array of probabilities of shape (15,15).
        initial_value (float): The model's evaluation of the initial position (-1 to 1).
        evaluated_moves (list): List of (move, value) tuples for the moves that were evaluated.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    board_tensor = raw_board_to_tensor(raw_board).to(device)
    
    # Initial forward pass to get policy and value for current position
    with torch.no_grad():
        policy_logits, value_pred = model(board_tensor)  # shapes: (1, 225), (1, 1)
        # Apply temperature scaling: divide logits by temperature
        scaled_logits = policy_logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)  # shape: (1, 225)
        initial_value = value_pred.item()  # Get the scalar value for initial position
    
    # Convert probabilities to a 15x15 numpy array
    probs_np = probs.cpu().numpy().reshape(15, 15)
    
    # Make sure we don't place a stone on an already occupied position
    mask = raw_board[:, :, 2] != 1.0  # Find non-empty cells
    probs_np[mask] = 0.0
    
    # To store details about evaluated moves
    evaluated_moves = []
    
    # For early moves (moves 0, 1, 2), use Top-K sampling for exploration
    if move_count < 3 and not use_value_eval:  # Only use this when not doing value evaluation
        # Determine K based on move count (more diversity early)
        k_values = [10, 7, 5]  # K for moves 0, 1, 2
        k = k_values[move_count]
        
        # Flatten probabilities
        flat_probs = probs_np.flatten()
        
        # Find indices of the top K probabilities
        top_k_indices = np.argsort(flat_probs)[-k:]
        
        # Get the corresponding probabilities
        top_k_probs = flat_probs[top_k_indices]
        
        # Normalize to sum to 1
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # Sample one index based on the probabilities
        sampled_idx = np.random.choice(top_k_indices, p=top_k_probs)
        
        # Convert to 2D coordinates
        row, col = sampled_idx // 15, sampled_idx % 15
        
        # Convert to 1-indexed coordinates
        top_move = (row + 1, col + 1)
        return top_move, probs_np, initial_value, []  # No evaluated moves in this case
    
    # If we're using value evaluation (Method 2) or past the early game exploration phase:
    if use_value_eval:
        # Get the top K legal moves from policy probabilities
        flat_probs = probs_np.flatten()
        # Get indices of all non-zero probabilities (legal moves)
        legal_indices = np.where(flat_probs > 0)[0]
        
        # If no legal moves found, return an error
        if len(legal_indices) == 0:
            raise ValueError("No legal moves available")
        
        # Sort by probability (ascending) and take the top K
        sorted_indices = legal_indices[np.argsort(flat_probs[legal_indices])]
        # Get the K highest probability moves
        top_indices = sorted_indices[-min(top_k, len(sorted_indices)):][::-1]  # Reverse to start with highest prob
        
        # Determine if we're black (True) or white (False)
        # Count stones to determine whose turn it is
        black_stones = np.sum(raw_board[:, :, 0])
        white_stones = np.sum(raw_board[:, :, 1])
        current_player = black_stones <= white_stones  # True if black's turn, False if white's turn
        
        # Initialize variables to track the best move
        best_move = None
        best_value_for_us = -float('inf')  # We want to maximize our value
        best_initial_prob = -1  # Track the probability of the best move for logging
        
        # Evaluate each of the top K moves
        for idx in top_indices:
            # Convert flat index to 2D coordinates (0-indexed)
            row, col = idx // 15, idx % 15
            # Convert to 1-indexed coordinates for place_stone
            candidate_move = (row + 1, col + 1)
            
            # Simulate making this move
            try:
                next_board = place_stone(raw_board, candidate_move, current_player=current_player)
                
                # Prepare the board tensor for evaluation
                next_board_tensor = raw_board_to_tensor(next_board).to(device)
                
                # Forward pass to get value prediction for the resulting position
                with torch.no_grad():
                    _, next_value_pred = model(next_board_tensor)
                    next_value = next_value_pred.item()
                
                # For the opponent's perspective, invert the value
                # (negative because opponent's win is our loss)
                opponent_expected_value = next_value
                our_expected_value = -opponent_expected_value  # We want opponent to have a bad position
                
                # Store the evaluated move data
                evaluated_moves.append({
                    'move': candidate_move,
                    'policy_prob': float(probs_np[row, col]),
                    'next_value': float(next_value),
                    'our_value': float(our_expected_value)
                })
                
                # If this move gives us a better expected outcome than previous best
                if our_expected_value > best_value_for_us:
                    best_value_for_us = our_expected_value
                    best_move = candidate_move
                    best_initial_prob = probs_np[row, col]
                    
            except ValueError as e:
                # Skip invalid moves - though this shouldn't happen if masking worked correctly
                continue
                
        # If we found a valid move through value evaluation
        if best_move is not None:
            # Return the best move, the original policy probabilities, and initial position value
            return best_move, probs_np, initial_value, evaluated_moves
    
    # Fallback: If value evaluation is disabled or unsuccessful, use regular argmax on policy
    row, col = np.unravel_index(np.argmax(probs_np), probs_np.shape)
    top_move = (row + 1, col + 1)
    return top_move, probs_np, initial_value, evaluated_moves

def visualize_board_and_prediction(raw_board, probabilities, top_move=None, value=None, evaluated_moves=None):
    """
    Visualize the current board state, the model's prediction, and win probability.
    
    Args:
        raw_board (np.array): Board in shape (15,15,3).
        probabilities (np.array): Prediction probabilities of shape (15,15).
        top_move (tuple, optional): The top predicted move as (row, col).
        value (float, optional): The model's evaluation of the position (-1 to 1).
        evaluated_moves (list, optional): List of dictionaries with moves evaluated by Method 2.
    """
    if evaluated_moves is None:
        plt.figure(figsize=(18, 6))
        num_plots = 3  # Board, probabilities, win probability
    else:
        plt.figure(figsize=(20, 10))
        num_plots = 4  # Board, probabilities, win probability, move evaluation
    
    # Plot the board
    plt.subplot(1, num_plots, 1)
    plt.imshow(np.ones((15, 15, 3)), cmap='Greys')
    
    # Draw grid lines
    for i in range(15):
        plt.axhline(i - 0.5, color='black', linewidth=1)
        plt.axvline(i - 0.5, color='black', linewidth=1)
    
    # Plot stones
    for i in range(15):
        for j in range(15):
            if raw_board[i, j, 0] == 1:  # Black stone
                plt.plot(j, i, 'o', markersize=15, color='black')
            elif raw_board[i, j, 1] == 1:  # White stone
                plt.plot(j, i, 'o', markersize=15, color='white', markeredgecolor='black')
    
    # Highlight the selected move if provided
    if top_move:
        row, col = top_move
        plt.plot(col-1, row-1, 'x', markersize=12, color='blue', markeredgewidth=3)
    
    # Highlight evaluated moves if provided
    if evaluated_moves:
        # Use different colors for evaluated moves to distinguish them
        for i, move_data in enumerate(evaluated_moves):
            move = move_data['move']
            row, col = move
            # Use a circular marker with different colors based on the value
            our_value = move_data['our_value']
            # Red for negative (bad for us), Green for positive (good for us)
            color = 'green' if our_value > 0 else 'red'
            # Size based on absolute value - bigger for more extreme values
            size = 8 + abs(our_value) * 5
            
            plt.plot(col-1, row-1, 'o', markersize=size, 
                     markerfacecolor='none', markeredgewidth=2, 
                     markeredgecolor=color, alpha=0.7)
            
            # Add value label next to the move
            plt.text(col-1+0.3, row-1, f"{our_value:.2f}", fontsize=8, color=color)
    
    plt.title('Current Board State')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.gca().invert_yaxis()
    
    # Plot the prediction probabilities
    plt.subplot(1, num_plots, 2)
    plt.imshow(probabilities, cmap='hot')
    
    if top_move:
        row, col = top_move
        plt.plot(col-1, row-1, 'x', markersize=10, color='blue')
    
    plt.colorbar(label='Probability')
    plt.title('Policy Probabilities')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.gca().invert_yaxis()
    
    # Plot win probability
    if value is not None:
        plt.subplot(1, num_plots, 3)
        
        # Create a pie chart showing win probability
        labels = ['Black wins', 'Draw', 'White wins']
        
        # Convert value from [-1, 1] range to probabilities (rough approximation)
        win_probs = [0, 0, 0]  # [black, draw, white]
        
        # Determine if the next player is Black or White
        black_stones = np.sum(raw_board[:, :, 0])
        white_stones = np.sum(raw_board[:, :, 1])
        is_black_turn = black_stones <= white_stones
        
        # Interpret the value for the current player
        # Map from [-1, 1] to win probability
        if value > 0:
            # Value > 0 means current player is favored
            win_prob = (value + 1) / 2  # Convert from [-1,1] to [0,1]
            draw_prob = 0.1  # Simplistic approach - always keep a small chance for draw
            loss_prob = 1 - win_prob - draw_prob
            
            # Ensure probabilities are valid
            if loss_prob < 0:
                loss_prob = 0
                draw_prob = 1 - win_prob
        else:
            # Value < 0 means opponent is favored
            loss_prob = (abs(value) + 1) / 2  # Convert from [-1,1] to [0,1]
            draw_prob = 0.1  # Simplistic approach
            win_prob = 1 - loss_prob - draw_prob
            
            # Ensure probabilities are valid
            if win_prob < 0:
                win_prob = 0
                draw_prob = 1 - loss_prob
        
        # Assign probabilities based on whose turn it is
        if is_black_turn:
            win_probs = [win_prob, draw_prob, loss_prob]
        else:
            win_probs = [loss_prob, draw_prob, win_prob]
        
        # Create the pie chart
        colors = ['black', 'gray', 'white']
        plt.pie(win_probs, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'black'})
        plt.title('Win Probability')
    
    # Plot evaluated moves comparison if provided
    if evaluated_moves and len(evaluated_moves) > 0:
        plt.subplot(1, num_plots, 4)
        
        # Sort evaluated moves by 'our_value' (best for us first)
        sorted_moves = sorted(evaluated_moves, key=lambda x: x['our_value'], reverse=True)
        
        moves = [f"({m['move'][0]},{m['move'][1]})" for m in sorted_moves]
        policy_probs = [m['policy_prob'] for m in sorted_moves]
        our_values = [m['our_value'] for m in sorted_moves]
        
        x = np.arange(len(moves))
        width = 0.35
        
        # Create a bar chart comparing policy probability vs. our value for each move
        ax = plt.gca()
        rects1 = ax.bar(x - width/2, policy_probs, width, label='Policy Probability')
        rects2 = ax.bar(x + width/2, our_values, width, label='Expected Value for Us')
        
        # Add some text for labels, title and custom x-axis tick labels
        ax.set_ylabel('Score')
        ax.set_title('Evaluated Moves Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(moves)
        ax.legend(loc='lower right')
        
        # Add values above the bars
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
                        
        for rect in rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
                        
        # Highlight the selected move
        if top_move is not None:
            top_move_str = f"({top_move[0]},{top_move[1]})"
            if top_move_str in moves:
                idx = moves.index(top_move_str)
                # Draw a rectangle around the selected move
                ax.get_xticklabels()[idx].set_color('blue')
                ax.get_xticklabels()[idx].set_weight('bold')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=300)
    plt.show()

def place_stone(raw_board, position, current_player=True):
    """
    Place a stone on the board at the given position.
    
    Args:
        raw_board (np.array): Board in shape (15,15,3).
        position (tuple): (row, col) coordinates (1-indexed).
        current_player (bool): True for a black stone, False for white.
        
    Returns:
        np.array: Updated board.
    """
    row, col = position
    # Convert to 0-indexed
    row_idx, col_idx = row - 1, col - 1
    
    # Check if position is valid and empty
    if not (0 <= row_idx < 15 and 0 <= col_idx < 15):
        raise ValueError(f"Position {position} is outside the board")
    if raw_board[row_idx, col_idx, 2] != 1.0:
        raise ValueError(f"Position {position} is already occupied")
    
    # Make a copy of the board
    new_board = raw_board.copy()
    
    # Place the stone
    channel = 0 if current_player else 1  # Black = channel 0, White = channel 1
    new_board[row_idx, col_idx, channel] = 1.0
    new_board[row_idx, col_idx, 2] = 0.0  # Mark as not empty
    
    return new_board

def predict_move_weighted(model, raw_board, temperature=1.0, alpha=0.7, move_count=0, device='cuda', top_k=8):
    """
    Predict the best move using a weighted combination of policy and value.
    
    Instead of using value head to evaluate the positions after moves have been made,
    this function directly combines policy probabilities with value estimates using
    a weighted average approach.
    
    Args:
        model: Loaded GomokuNet model.
        raw_board (np.array): Board in shape (15,15,3).
        temperature (float): Temperature scaling factor for policy.
        alpha (float): Weight for policy (1-alpha is the weight for value),
                       typically between 0.5 and 0.9.
        move_count (int): Current move count in the game.
        device (str): 'cuda' or 'cpu'.
        top_k (int): Number of top policy moves to evaluate.
        
    Returns:
        top_move (tuple): (row, col) for the selected move (1-indexed).
        probabilities (np.array): Array of probabilities of shape (15,15).
        initial_value (float): The model's evaluation of the position (-1 to 1).
        evaluated_moves (list): List of dictionaries with evaluated moves.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    board_tensor = raw_board_to_tensor(raw_board).to(device)
    
    # Initial forward pass to get policy and value for current position
    with torch.no_grad():
        policy_logits, value_pred = model(board_tensor)
        scaled_logits = policy_logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        initial_value = value_pred.item()
    
    # Convert probabilities to a 15x15 numpy array
    probs_np = probs.cpu().numpy().reshape(15, 15)
    
    # Make sure we don't place a stone on an already occupied position
    mask = raw_board[:, :, 2] != 1.0  # Find non-empty cells
    probs_np[mask] = 0.0
    
    # Get the top K legal moves from policy probabilities
    flat_probs = probs_np.flatten()
    legal_indices = np.where(flat_probs > 0)[0]
    
    if len(legal_indices) == 0:
        raise ValueError("No legal moves available")
    
    # Sort by probability and take the top K
    sorted_indices = legal_indices[np.argsort(flat_probs[legal_indices])]
    top_indices = sorted_indices[-min(top_k, len(sorted_indices)):][::-1]
    
    # Determine the current player
    black_stones = np.sum(raw_board[:, :, 0])
    white_stones = np.sum(raw_board[:, :, 1])
    current_player = black_stones <= white_stones  # True if black's turn
    
    # Initialize variables for tracking the best move
    best_move = None
    best_combined_score = -float('inf')
    evaluated_moves = []
    
    # Evaluate each of the top K moves
    for idx in top_indices:
        row, col = idx // 15, idx % 15
        candidate_move = (row + 1, col + 1)
        policy_score = probs_np[row, col]
        
        try:
            # Simulate making this move
            next_board = place_stone(raw_board, candidate_move, current_player=current_player)
            next_board_tensor = raw_board_to_tensor(next_board).to(device)
            
            # Get value for the resulting position
            with torch.no_grad():
                _, next_value_pred = model(next_board_tensor)
                next_value = next_value_pred.item()
            
            # Negate the value (from opponent's perspective)
            our_expected_value = -next_value
            
            # Calculate combined score using weighted average
            combined_score = alpha * policy_score + (1-alpha) * (our_expected_value + 1)/2
            
            # Store evaluation data
            evaluated_moves.append({
                'move': candidate_move,
                'policy_prob': float(policy_score),
                'value': float(our_expected_value),
                'combined_score': float(combined_score)
            })
            
            # Update best move if this one has higher combined score
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_move = candidate_move
                
        except ValueError:
            continue
    
    # Fallback to policy if no valid moves found
    if best_move is None:
        row, col = np.unravel_index(np.argmax(probs_np), probs_np.shape)
        best_move = (row + 1, col + 1)
        
    return best_move, probs_np, initial_value, evaluated_moves

def predict_move_with_weighted_details(model, raw_board, temperature=1.0, alpha=0.7, move_count=0, device='cuda', top_k=8):
    """
    Wrapper for predict_move_weighted that ensures compatibility with the GUI.
    
    This is a simpler version that returns the same output format as predict_move_with_details.
    """
    return predict_move_weighted(model, raw_board, temperature, alpha, move_count, device, top_k)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gomoku Inference Script")
    parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling factor for logits")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the board and predictions")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = load_model(args.model_path, device=device)
    
    # Example: Create an empty board
    raw_board = np.zeros((15, 15, 3), dtype=np.float32)
    raw_board[:, :, 2] = 1.0  # Initialize all positions as empty
    
    # Example: Let's simulate a few moves
    try:
        # Place some stones (simulating a game in progress)
        # For example, place black stones at (8,8) and (8,9)
        raw_board = place_stone(raw_board, (8, 8), current_player=True)
        raw_board = place_stone(raw_board, (8, 9), current_player=False)
        
        # Predict the best move
        top_move, probs, value = predict_move(model, raw_board, temperature=args.temperature, device=device)
        print("Top predicted move (1-indexed):", top_move)
        print(f"Position evaluation: {value:.4f} ({'+' if value > 0 else ''}{value:.2f})")
        
        # Visualize if requested
        if args.visualize:
            visualize_board_and_prediction(raw_board, probs, top_move, value)
            
        # Place the predicted move on the board (alternating colors)
        raw_board = place_stone(raw_board, top_move, current_player=True)
        
        # Predict another move
        top_move, probs, value = predict_move(model, raw_board, temperature=args.temperature, device=device)
        print("Next top predicted move (1-indexed):", top_move)
        print(f"Position evaluation: {value:.4f} ({'+' if value > 0 else ''}{value:.2f})")
        
        if args.visualize:
            visualize_board_and_prediction(raw_board, probs, top_move, value)
            
    except Exception as e:
        print(f"Error: {e}")
