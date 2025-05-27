import argparse
import time
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import threading
import datetime
import pygame
import traceback

# Add parent directory to path so we can import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GomokuNet, ResidualBlock


BOARD_SIZE = 15
WIN_LENGTH = 5  
CELL_SIZE = 40  
BOARD_MARGIN = 40  


BOARD_COLOR = (220, 180, 100)  
LINE_COLOR = (0, 0, 0)  
BLACK_STONE = (0, 0, 0)  
WHITE_STONE = (255, 255, 255)  
RED_HIGHLIGHT = (255, 0, 0)  

class GomokuGame:
    """Class representing a Gomoku game state."""
    
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        self.board[:, :, 2] = 1.0  
        self.current_player = True  
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.board_history = []
        self.move_count = 0
        self.winning_line = []  
        
    def make_move(self, row, col):
        """Make a move at the specified position.
        
        Args:
            row: 1-based row index
            col: 1-based column index
            
        Returns:
            True if move was legal and successful, False otherwise
        """
        r, c = row - 1, col - 1
        
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.board[r, c, 2] != 1.0:  
            return False
            
        channel = 0 if self.current_player else 1  
        self.board[r, c, channel] = 1.0
        self.board[r, c, 2] = 0.0  
        self.move_history.append((row, col))
        self.board_history.append(np.copy(self.board))
        self.move_count += 1
        
        if self.check_win(r, c):
            self.game_over = True
            self.winner = self.current_player
            player_name = "Black" if self.current_player else "White"
            print(f"Player {player_name} wins at move {self.move_count}!")
            return True
            
        if np.sum(self.board[:, :, 2]) == 0:  
            self.game_over = True
            self.winner = None  
            print(f"Game ended in a draw after {self.move_count} moves!")
            return True
            
        self.current_player = not self.current_player
        return True
    
    def check_win(self, row, col):
        """Check if the last move at (row, col) resulted in a win.
        Uses a robust algorithm that checks in each direction separately.
        
        Args:
            row: 0-indexed row
            col: 0-indexed column
            
        Returns:
            True if the move resulted in a win, False otherwise
        """
        channel = 0 if self.current_player else 1
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  
            line = [(row, col)]
            
            for i in range(1, WIN_LENGTH):
                r, c = row + i*dr, col + i*dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c, channel] == 1.0:
                    count += 1
                    line.append((r, c))
                else:
                    break
                    
            for i in range(1, WIN_LENGTH):
                r, c = row - i*dr, col - i*dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c, channel] == 1.0:
                    count += 1
                    line.append((r, c))
                else:
                    break
            
            if count >= WIN_LENGTH:
                self.winning_line = line
                return True
                
        return False
        
    def get_legal_moves(self):
        """Return list of legal moves as (row, col) tuples in 1-indexed form."""
        if self.game_over:
            return []
            
        legal_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c, 2] == 1.0:  
                    legal_moves.append((r+1, c+1)) 
                    
        return legal_moves
        
    def get_result(self):
        """Return game result: 1 for black win, -1 for white win, 0 for draw, None if ongoing."""
        if not self.game_over:
            return None
        if self.winner is None:
            return 0  
        return 1 if self.winner else -1  

def raw_board_to_tensor(raw_board):
    """Convert a raw board (numpy array) to a torch tensor for neural network input."""
    board_tensor = torch.tensor(raw_board, dtype=torch.float32)
    board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0) 
    return board_tensor

# ========== Neural Network Component Replacements for Ablation Tests ==========

class ChannelAttentionReplacement(torch.nn.Module):
    
    def __init__(self, channels=64, reduction=16):
        super().__init__()
        self.channels = channels
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = self.avg_pool(x)
        return self.fc(avg_out)

class SPATREPL(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(avg_out))
        return attention

class NNLREPL(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = torch.nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.key = torch.nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.value = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = torch.nn.functional.softmax(energy, dim=2)
        
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return x + self.gamma * out

class MSREPL(torch.nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def build_ablated_model(base_model, ablation_config=None):
    """Build a new model with certain components ablated.
    
    Args:
        base_model: Original model to copy structure from
        ablation_config: Dictionary with keys for components to ablate
                      {'no_channel_attention': True, 'no_spatial_attention': True, etc.}
    
    Returns:
        A new GomokuNet model with specified components replaced
    """
    device = next(base_model.parameters()).device
    
    ablated_model = GomokuNet().to(device)
    
    if not ablation_config:
        ablated_model.load_state_dict(base_model.state_dict())
        print("No ablation: created exact copy of base model")
        return ablated_model
    
    ablated_model.load_state_dict(base_model.state_dict())
    
    print(f"Applying ablation config: {ablation_config}")
    ablation_applied = False
    
    if ablation_config.get('no_channel_attention', False) or ablation_config.get('no_cbam', False):
        channels = ablated_model.channel_attention.fc[0].in_channels
        ablated_model.channel_attention = ChannelAttentionReplacement(channels=channels).to(device)
        print(f"Replaced channel attention  (channels={channels})")
        ablation_applied = True
    
    if ablation_config.get('no_spatial_attention', False) or ablation_config.get('no_cbam', False):
        kernel_size = ablated_model.spatial_attention.conv.kernel_size[0]
        ablated_model.spatial_attention = SPATREPL(kernel_size=kernel_size).to(device)
        print(f"Replaced spatial attention  (kernel_size={kernel_size})")
        ablation_applied = True
    
    if ablation_config.get('no_non_local', False):
        in_channels = ablated_model.non_local.g.in_channels
        ablated_model.non_local = NNLREPL(in_channels).to(device)
        print(f"Replaced non-local block  (in_channels={in_channels})")
        ablation_applied = True
        
    if ablation_config.get('no_multi_scale', False):
        channels = ablated_model.multi_scale.convs[0].out_channels
        ablated_model.multi_scale = MSREPL(channels=channels).to(device)
        print(f"Replaced multi-scale block with single convolution (channels={channels})")
        ablation_applied = True
    
    if ablation_config.get('no_dilated_conv', False):
        channels = ablated_model.resblock1.conv1.out_channels
        dropout_rates = [
            ablated_model.resblock1.dropout.p,
            ablated_model.resblock2.dropout.p,
            ablated_model.resblock3.dropout.p
        ]
        ablated_model.resblock1 = ResidualBlock(channels, dilation=1, dropout_rate=dropout_rates[0]).to(device)
        ablated_model.resblock2 = ResidualBlock(channels, dilation=1, dropout_rate=dropout_rates[1]).to(device)
        ablated_model.resblock3 = ResidualBlock(channels, dilation=1, dropout_rate=dropout_rates[2]).to(device)
        print(f"Replaced dilated convolutions with regular convolutions (channels={channels})")
        ablation_applied = True
    
    if not ablation_applied:
        print("Warning: Ablation config provided but no components were ablated")
    
    ablated_model.eval()
    
    print(f"Created ablated model with {sum(p.numel() for p in ablated_model.parameters())} parameters")
    return ablated_model

# ========== Board Visualization Functions ==========

def init_pygame_display():
    """Initialize the pygame display for board visualization."""
    pygame.init()
    pygame.display.set_caption("Gomoku Ablation Battle")
    board_width = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN
    board_height = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 50  # Extra space for title
    board_display = pygame.display.set_mode((board_width, board_height))
    return board_display

def draw_board(display, board, last_move=None, winning_line=None, title=None):
    """Draw the Gomoku board with stones."""
    if display is None:
        return
    
    display.fill(BOARD_COLOR)
    
    for i in range(BOARD_SIZE):
        pygame.draw.line(
            display, 
            LINE_COLOR, 
            (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN),
            (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE),
            1 if i > 0 and i < BOARD_SIZE - 1 else 2
        )
        
        pygame.draw.line(
            display, 
            LINE_COLOR, 
            (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
            (BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
            1 if i > 0 and i < BOARD_SIZE - 1 else 2
        )
    
    star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
    for x, y in star_points:
        pygame.draw.circle(
            display,
            LINE_COLOR,
            (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
            4
        )
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i, j, 0] == 1.0:  # Black stone
                pygame.draw.circle(
                    display,
                    BLACK_STONE,
                    (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                    CELL_SIZE // 2 - 2
                )
            elif board[i, j, 1] == 1.0:  # White stone
                pygame.draw.circle(
                    display,
                    WHITE_STONE,
                    (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                    CELL_SIZE // 2 - 2,
                    0  # Filled circle
                )
                pygame.draw.circle(
                    display,
                    LINE_COLOR,
                    (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                    CELL_SIZE // 2 - 2,
                    1  # Outline
                )
    
    if last_move is not None:
        row, col = last_move
        pygame.draw.circle(
            display,
            RED_HIGHLIGHT,
            (BOARD_MARGIN + (col-1) * CELL_SIZE, BOARD_MARGIN + (row-1) * CELL_SIZE),
            5,
            2  # Width of circle
        )
    
    if winning_line is not None and len(winning_line) >= WIN_LENGTH:
        for i in range(len(winning_line) - 1):
            r1, c1 = winning_line[i]
            r2, c2 = winning_line[i + 1]
            pygame.draw.line(
                display,
                RED_HIGHLIGHT,
                (BOARD_MARGIN + c1 * CELL_SIZE, BOARD_MARGIN + r1 * CELL_SIZE),
                (BOARD_MARGIN + c2 * CELL_SIZE, BOARD_MARGIN + r2 * CELL_SIZE),
                3  # Line width
            )
    
    if title:
        font = pygame.font.SysFont(None, 28)
        title_surface = font.render(title, True, LINE_COLOR)
        display.blit(
            title_surface,
            (BOARD_MARGIN, BOARD_MARGIN + BOARD_SIZE * CELL_SIZE + 10)
        )
    
    pygame.display.flip()

def predict_move(model, raw_board, temperature=1.0, move_count=0, device='cuda'):
    """Predict the best move for a given board state using the model.
    
    Args:
        model: Neural network model
        raw_board: Board state as numpy array (15,15,3)
        temperature: Temperature for softmax sampling (higher = more random)
        move_count: Current move count in the game
        device: Device to run inference on
        
    Returns:
        top_move: (row, col) tuple with 1-indexed coordinates
        probs: Probability distribution over moves
        value: Value prediction (-1 to 1)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    board_tensor = raw_board_to_tensor(raw_board).to(device)
    
    with torch.no_grad():
        policy_logits, value_pred = model(board_tensor)
        scaled_logits = policy_logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        value = value_pred.item()
    
    probs_np = probs.cpu().numpy().reshape(BOARD_SIZE, BOARD_SIZE)
    mask = raw_board[:, :, 2] != 1.0  
    probs_np[mask] = 0.0
    
    if np.sum(probs_np) <= 0:
        legal_positions = np.where(raw_board[:, :, 2] == 1.0)
        if len(legal_positions[0]) > 0:
            idx = np.random.randint(len(legal_positions[0]))
            row, col = legal_positions[0][idx], legal_positions[1][idx]
            return (row + 1, col + 1), probs_np, value
        else:
            return (BOARD_SIZE // 2 + 1, BOARD_SIZE // 2 + 1), probs_np, value
    
    probs_np = probs_np / np.sum(probs_np)
    
    if move_count < 10 or temperature > 0.2:
        k = max(3, min(10, int(5 * temperature)))
        flat_probs = probs_np.flatten()
        indices = np.argsort(flat_probs)[-k:]  
        
        top_k_probs = flat_probs[indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)  
        
        chosen_idx = np.random.choice(indices, p=top_k_probs)
        
        row, col = chosen_idx // BOARD_SIZE, chosen_idx % BOARD_SIZE
        
        top_move = (row + 1, col + 1)
        return top_move, probs_np, value
    
    row, col = np.unravel_index(np.argmax(probs_np), probs_np.shape)
    
    top_move = (row + 1, col + 1)
    return top_move, probs_np, value

def play_game_with_temps(model_black, model_white, temperature_black=0.8, temperature_white=0.8, max_moves=225, 
              device='cuda', visualize=False, delay=0.5):
    """Simulate a full game between two models with different temperature settings.
    
    Args:
        model_black: Model playing as black
        model_white: Model playing as white
        temperature_black: Temperature for black's move sampling
        temperature_white: Temperature for white's move sampling
        max_moves: Maximum number of moves before declaring a draw
        device: Device to run inference on
        visualize: Whether to visualize the game
        delay: Delay between moves when visualizing (seconds)
        
    Returns:
        Dictionary with game results
    """
    game = GomokuGame()
    models = {True: model_black, False: model_white}  
    temperatures = {True: temperature_black, False: temperature_white}  
    
    black_policy_confidence = []
    white_policy_confidence = []
    
    illegal_move_attempts = {True: 0, False: 0} 
    consecutive_illegal_moves = {True: 0, False: 0}  
    
    board_display = None
    if visualize:
        board_display = init_pygame_display()
        draw_board(
            board_display, 
            game.board, 
            title="Game in Progress: Black's turn"
        )
    
    while not game.game_over and game.move_count < max_moves:
        current_model = models[game.current_player]
        current_player_name = "Black" if game.current_player else "White"
        current_temp = temperatures[game.current_player]
        
        if visualize:
            draw_board(
                board_display,
                game.board,
                title=f"Game in Progress: {current_player_name}'s turn (move {game.move_count+1})"
            )
        
        try:
            legal_moves = game.get_legal_moves()
            print(f"{current_player_name} has {len(legal_moves)} legal moves available")
            
            top_move, probs, value = predict_move(
                current_model, 
                game.board, 
                temperature=current_temp,  
                move_count=game.move_count, 
                device=device
            )
            
            top_prob = np.max(probs)
            if game.current_player:  # Black
                black_policy_confidence.append(top_prob)
            else:  # White
                white_policy_confidence.append(top_prob)
                
            print(f"{current_player_name} model predicted move: {top_move}")
            success = game.make_move(top_move[0], top_move[1])
            
            if success:
                consecutive_illegal_moves[game.current_player] = 0
                
                if visualize:
                    draw_board(
                        board_display,
                        game.board,
                        last_move=top_move,
                        winning_line=game.winning_line if game.game_over else None,
                        title=f"Move {game.move_count}: {current_player_name} played at {top_move}"
                    )
                    time.sleep(delay)  
            else:
                print(f"Model made illegal move: {top_move}")
                
                illegal_move_attempts[game.current_player] += 1
                consecutive_illegal_moves[game.current_player] += 1
                
                if consecutive_illegal_moves[game.current_player] >= 5:
                    print(f"ERROR: {current_player_name} model made 5 consecutive illegal moves. Stopping game.")
                    game.game_over = True
                    game.winner = not game.current_player
                    break
                
                if legal_moves:
                    random_move = legal_moves[np.random.randint(len(legal_moves))]
                    print(f"Selecting random legal move: {random_move}")
                    game.make_move(random_move[0], random_move[1])
                    
                    if visualize:
                        draw_board(
                            board_display,
                            game.board,
                            last_move=random_move,
                            winning_line=game.winning_line if game.game_over else None,
                            title=f"Move {game.move_count}: {current_player_name} played RANDOM at {random_move}"
                        )
                        time.sleep(delay)
                else:
                    print("No legal moves available. Game is a draw.")
                    game.game_over = True
                    game.winner = None
                    
        except Exception as e:
            print(f"Error during model prediction: {str(e)}")
            traceback.print_exc()  
            
            legal_moves = game.get_legal_moves()
            if legal_moves:
                random_move = legal_moves[np.random.randint(len(legal_moves))]
                print(f"Error occurred, selecting random legal move: {random_move}")
                game.make_move(random_move[0], random_move[1])
                
                if visualize:
                    draw_board(
                        board_display,
                        game.board,
                        last_move=random_move,
                        winning_line=game.winning_line if game.game_over else None,
                        title=f"Move {game.move_count}: {current_player_name} played RANDOM at {random_move}"
                    )
                    time.sleep(delay)
            else:
                game.game_over = True
                game.winner = None
    
    if visualize:
        result_text = "Draw"
        if game.winner is not None:
            result_text = "Black wins" if game.winner else "White wins"
        
        draw_board(
            board_display,
            game.board,
            winning_line=game.winning_line,
            title=f"Game over: {result_text} in {game.move_count} moves"
        )
        time.sleep(2.0)  
    
    result = {
        'result': 1 if game.winner else (-1 if game.winner is not None else 0),
        'moves': game.move_count,
        'black_confidence': np.mean(black_policy_confidence) if black_policy_confidence else 0,
        'white_confidence': np.mean(white_policy_confidence) if white_policy_confidence else 0,
        'black_illegal_moves': illegal_move_attempts[True],
        'white_illegal_moves': illegal_move_attempts[False],
    }
    
    winner_str = "Draw"
    if game.winner is not None:
        winner_str = "Black" if game.winner else "White"
    print(f"Player {winner_str} wins at move {game.move_count}!")
    print(f"Illegal move attempts - Black: {illegal_move_attempts[True]}, White: {illegal_move_attempts[False]}")
    
    return result

def play_game(model_black, model_white, temperature=0.8, max_moves=225, 
              device='cuda', visualize=False, delay=0.5):
    """Simulate a full game between two models.
    
    Args:
        model_black: Model playing as black
        model_white: Model playing as white
        temperature: Temperature for move sampling
        max_moves: Maximum number of moves before declaring a draw
        device: Device to run inference on
        visualize: Whether to visualize the game
        delay: Delay between moves when visualizing (seconds)
        
    Returns:
        Dictionary with game results
    """
    return play_game_with_temps(
        model_black=model_black,
        model_white=model_white,
        temperature_black=temperature,
        temperature_white=temperature,
        max_moves=max_moves,
        device=device,
        visualize=visualize,
        delay=delay
    )

# ========== Component Ablation Testing Framework ==========

class ComponentAblationBattle:
    """Framework for testing the impact of different neural network components."""
    
    def __init__(self, model_path, output_dir="ablation_results", device='cuda'):
        """Initialize the ablation battle framework.
        
        Args:
            model_path: Path to the trained model checkpoint
            output_dir: Directory to save results
            device: Device to run on (cuda or cpu)
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading base model from {model_path}...")
        self.base_model = load_model(model_path, device=self.device)
        
        try:
            dummy_input = torch.zeros(1, 3, BOARD_SIZE, BOARD_SIZE, device=self.device)
            with torch.no_grad():
                policy, value = self.base_model(dummy_input)
                print(f"Model validation successful - policy output shape: {policy.shape}, value: {value.item():.4f}")
            
            empty_board = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
            empty_board[:, :, 2] = 1.0  # Set "empty" channel to 1.0
            top_move, probs, value = predict_move(
                self.base_model, 
                empty_board, 
                temperature=0.3,
                device=self.device
            )
            print(f"Test prediction on empty board: move={top_move}, value={value:.4f}")
            
        except Exception as e:
            print(f"ERROR: Model validation failed: {str(e)}")
            print("The model may not be properly loaded or may have structural issues.")
            traceback.print_exc()
            print("Continuing with the model, but results may be unreliable.")
        
    def create_ablated_model(self, ablation_config=None):
        """Create an ablated model based on the configuration."""
        return build_ablated_model(self.base_model, ablation_config)
    
    def run_ablation_test(self, ablation_config, name, num_games=20, temperature=0.6, visualize=False):
        """Run a battle between the base model and an ablated version.
        
        Args:
            ablation_config: Dictionary of components to ablate
            name: Name for this ablation test
            num_games: Number of games to play
            temperature: Temperature for move sampling (higher values = more randomness)
            visualize: Whether to visualize the games
            
        Returns:
            Dictionary with battle results
        """
        print(f"\n{'-'*50}")
        print(f"Running ablation test: {name}")
        print(f"{'-'*50}")
        
        ablated_model = self.create_ablated_model(ablation_config)
        
        results = {
            'ablation_name': name,
            'base_wins_as_black': 0,
            'ablated_wins_as_black': 0,
            'base_wins_as_white': 0,
            'ablated_wins_as_white': 0,
            'draws': 0,
            'avg_game_length': 0,
            'base_avg_confidence': 0,
            'ablated_avg_confidence': 0,
            'games': [],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        total_moves = 0
        base_confidence_black = []
        base_confidence_white = []
        ablated_confidence_black = []
        ablated_confidence_white = []
        
        games_per_side = num_games // 2
        print(f"Playing {games_per_side} games with base model as Black vs ablated model as White")
        
        base_temp = temperature
        ablated_temp = temperature * 1.2 
        
        for i in tqdm(range(games_per_side)):
            if i % 2 == 0:
                game_temp_base = base_temp * 1.5
                game_temp_ablated = ablated_temp * 1.5
            else:
                game_temp_base = base_temp
                game_temp_ablated = ablated_temp  
            game_result = play_game_with_temps(
                self.base_model, ablated_model,
                temperature_black=game_temp_base,
                temperature_white=game_temp_ablated,
                device=self.device,
                visualize=visualize,
                delay=0.5 if visualize else 0
            )
            
            total_moves += game_result['moves']
            base_confidence_black.extend([game_result['black_confidence']])
            ablated_confidence_white.extend([game_result['white_confidence']])
            
            if game_result['result'] == 1:  # Black (base) wins
                results['base_wins_as_black'] += 1
            elif game_result['result'] == -1:  # White (ablated) wins
                results['ablated_wins_as_white'] += 1
            else:  # Draw
                results['draws'] += 1
            results['games'].append({
                'black': 'base',
                'white': 'ablated',
                'result': game_result['result'],
                'moves': game_result['moves']
            })
            
        print(f"Playing {games_per_side} games with ablated model as Black vs base model as White")
        
        for i in tqdm(range(games_per_side)):
            if i % 2 == 0:
                game_temp_base = base_temp * 1.5
                game_temp_ablated = ablated_temp * 1.5
            else:
                game_temp_base = base_temp
                game_temp_ablated = ablated_temp
            
            game_result = play_game_with_temps(
                ablated_model, self.base_model,
                temperature_black=game_temp_ablated,
                temperature_white=game_temp_base,
                device=self.device,
                visualize=visualize,
                delay=0.5 if visualize else 0
            )
            
            total_moves += game_result['moves']
            ablated_confidence_black.extend([game_result['black_confidence']])
            base_confidence_white.extend([game_result['white_confidence']])
            
            if game_result['result'] == 1:  # Black (ablated) wins
                results['ablated_wins_as_black'] += 1
            elif game_result['result'] == -1:  # White (base) wins
                results['base_wins_as_white'] += 1
            else:  # Draw
                results['draws'] += 1
                
            results['games'].append({
                'black': 'ablated',
                'white': 'base',
                'result': game_result['result'],
                'moves': game_result['moves']
            })
            
        results['avg_game_length'] = total_moves / num_games
        
        if base_confidence_black and base_confidence_white:
            results['base_avg_confidence'] = np.mean(base_confidence_black + base_confidence_white)
        
        if ablated_confidence_black and ablated_confidence_white:
            results['ablated_avg_confidence'] = np.mean(ablated_confidence_black + ablated_confidence_white)
        
        # Calculate total wins
        results['base_total_wins'] = results['base_wins_as_black'] + results['base_wins_as_white']
        results['ablated_total_wins'] = results['ablated_wins_as_black'] + results['ablated_wins_as_white']
        
        # Calculate win rates
        results['base_win_rate'] = results['base_total_wins'] / num_games
        results['ablated_win_rate'] = results['ablated_total_wins'] / num_games
        results['draw_rate'] = results['draws'] / num_games
        
        # Approximate Elo difference
        if results['base_win_rate'] > 0 and results['ablated_win_rate'] > 0:
            elo_diff = 400 * np.log10(results['base_win_rate'] / results['ablated_win_rate'])
            results['estimated_elo_diff'] = elo_diff
        
        # Print summary
        self.print_battle_summary(results, "Base Model", f"Ablated ({name})")
        
        # Save results
        self.save_results(results, name)
        
        # Create visualization
        self.create_ablation_visualization(results, name, results['timestamp'])
        
        return results 

    def print_battle_summary(self, results, model_a_name, model_b_name):
        """Print a formatted summary of battle results."""
        print("\n" + "="*50)
        print(f"BATTLE RESULTS: {model_a_name} vs {model_b_name}")
        print("="*50)
        
        # Print general stats
        print(f"Total Games: {len(results['games'])}")
        print(f"Wins - {model_a_name}: {results['base_total_wins']} ({results['base_win_rate']:.2%})")
        print(f"Wins - {model_b_name}: {results['ablated_total_wins']} ({results['ablated_win_rate']:.2%})")
        print(f"Draws: {results['draws']} ({results['draw_rate']:.2%})")
        
        # Print detailed stats
        print(f"\n{model_a_name} wins as Black: {results['base_wins_as_black']}")
        print(f"{model_a_name} wins as White: {results['base_wins_as_white']}")
        print(f"{model_b_name} wins as Black: {results['ablated_wins_as_black']}")
        print(f"{model_b_name} wins as White: {results['ablated_wins_as_white']}")
        
        # Print confidence and game length
        print(f"\nAverage Game Length: {results['avg_game_length']:.1f} moves")
        print(f"{model_a_name} Average Confidence: {results['base_avg_confidence']:.4f}")
        print(f"{model_b_name} Average Confidence: {results['ablated_avg_confidence']:.4f}")
        
        # Print Elo difference if available
        if 'estimated_elo_diff' in results:
            print(f"\nEstimated Elo Difference: {results['estimated_elo_diff']:.1f}")
        
        print("="*50)
        
        # Print conclusion
        if results['base_total_wins'] > results['ablated_total_wins']:
            winner = model_a_name
            win_margin = results['base_total_wins'] - results['ablated_total_wins']
            win_percentage = results['base_win_rate']
            print(f"WINNER: {winner} won by {win_margin} games ({win_percentage:.2%} win rate)")
            print(f"This suggests that ablating these components REDUCES performance.")
        elif results['ablated_total_wins'] > results['base_total_wins']:
            winner = model_b_name
            win_margin = results['ablated_total_wins'] - results['base_total_wins']
            win_percentage = results['ablated_win_rate']
            print(f"WINNER: {winner} won by {win_margin} games ({win_percentage:.2%} win rate)")
            print(f"This suggests that ablating these components IMPROVES performance or adds noise.")
        else:
            print("The battle ended in a DRAW. The ablated components may not significantly impact performance.")
            
        print("="*50 + "\n")
    

    
    def create_ablation_visualization(self, results, ablation_name, timestamp):
        """Create visualizations of battle results."""
        plt.figure(figsize=(12, 8))
        
        # Data for the grouped bar chart
        models = ["Base Model", f"Ablated ({ablation_name})"]
        wins_as_black = [results['base_wins_as_black'], results['ablated_wins_as_black']]
        wins_as_white = [results['base_wins_as_white'], results['ablated_wins_as_white']]
        
        # Set width of bars
        barWidth = 0.3
        r1 = np.arange(len(models))
        r2 = [x + barWidth for x in r1]
        
        # Create grouped bars
        plt.bar(r1, wins_as_black, width=barWidth, label='Wins as Black', color='black')
        plt.bar(r2, wins_as_white, width=barWidth, label='Wins as White', color='lightgray', edgecolor='black')
        
        # Add values above bars
        for i, v in enumerate(wins_as_black):
            plt.text(i, v + 0.1, str(v), ha='center')
            
        for i, v in enumerate(wins_as_white):
            plt.text(i + barWidth, v + 0.1, str(v), ha='center')
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('Number of Wins')
        plt.title(f'Ablation Test: {ablation_name}')
        plt.xticks([r + barWidth/2 for r in range(len(models))], models)
        plt.legend()
        
        # Save the figure
        plt.savefig(f"{self.output_dir}/wins_by_color_{ablation_name}_{timestamp}.png", dpi=150)
        
        # Create pie chart of overall results
        plt.figure(figsize=(10, 8))
        
        # Data for pie chart
        labels = ["Base Model Wins", f"Ablated ({ablation_name}) Wins", "Draws"]
        sizes = [results['base_total_wins'], results['ablated_total_wins'], results['draws']]
        colors = ['#3498db', '#e74c3c', '#95a5a6']  # Blue, Red, Gray
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Overall Results')
        
        plt.savefig(f"{self.output_dir}/overall_results_{ablation_name}_{timestamp}.png", dpi=150)
        
        plt.close('all')
    
    def run_all_ablation_tests(self, num_games=100, temperature=0.6, visualize=False):
        """Run all possible ablation tests.
        
        Args:
            num_games: Number of games for each test
            temperature: Temperature for move sampling (higher = more random)
            visualize: Whether to visualize games
            
        Returns:
            Dictionary mapping ablation names to results
        """
        # Define the ablation configurations to test
        ablation_configs = {
            "no_cbam": {'no_cbam': True},
            "no_channel_attention": {'no_channel_attention': True},
            "no_spatial_attention": {'no_spatial_attention': True},
            "no_non_local": {'no_non_local': True},
            "no_multi_scale": {'no_multi_scale': True},
            "no_dilated_conv": {'no_dilated_conv': True}
        }
        
        # Dictionary to store all results
        all_results = {}
        
        # Run each ablation test
        for name, config in ablation_configs.items():
            print(f"\nRunning ablation test for {name}...")
            results = self.run_ablation_test(
                ablation_config=config,
                name=name,
                num_games=num_games,
                temperature=temperature,
                visualize=visualize
            )
            all_results[name] = results
        
        # Generate a comprehensive report
        self.generate_ablation_report(all_results)
        
        return all_results
    

def load_model(model_path, device='cuda'):
    """Load a GomokuNet model from a checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = GomokuNet()
    
    try:
        # Try loading with strict=True first
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Warning: Error loading model with strict mode: {str(e)}")
        print("Attempting to load with strict=False...")
        # Try loading with strict=False
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Model loaded with strict=False, some weights may be missing or not initialized properly.")
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Verify model was loaded properly
    print(f"Model loaded from {model_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Gomoku Neural Network Component Ablation Testing")
    parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                      help="Path to the trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="ablation_results",
                      help="Directory to save results and visualizations")
    parser.add_argument("--ablation", type=str, default="all",
                      choices=["all", "no_cbam", "no_channel_attention", "no_spatial_attention", 
                               "no_non_local", "no_multi_scale", "no_dilated_conv"],
                      help="Specific ablation test to run (default: all)")
    parser.add_argument("--num_games", type=int, default=100,
                      help="Number of games to play for each ablation test")
    parser.add_argument("--temperature", type=float, default=0.6,
                      help="Temperature for move sampling (higher = more random, 0.6 recommended)")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize games using pygame")
    parser.add_argument("--device", type=str, default="cuda",
                      choices=["cuda", "cpu"],
                      help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Create the ablation battle framework
    ablation_battle = ComponentAblationBattle(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run the specified ablation test(s)
    if args.ablation == "all":
        ablation_battle.run_all_ablation_tests(
            num_games=args.num_games,
            temperature=args.temperature,
            visualize=args.visualize
        )
    else:
        # Create the appropriate ablation config
        ablation_config = {args.ablation: True}
        
        # Handle special case for no_cbam
        if args.ablation == "no_cbam":
            ablation_config = {'no_cbam': True}
        
        # Run the single ablation test
        ablation_battle.run_ablation_test(
            ablation_config=ablation_config,
            name=args.ablation,
            num_games=args.num_games,
            temperature=args.temperature,
            visualize=args.visualize
        )
    
    print("Ablation testing completed successfully.")
    
if __name__ == "__main__":
    main() 