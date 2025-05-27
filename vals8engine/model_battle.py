import argparse
import time
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import pygame

from model import GomokuNet
from inference import raw_board_to_tensor, predict_move

# Constants
BOARD_SIZE = 15
WIN_LENGTH = 5  # Standard Gomoku rule is 5-in-a-row
CELL_SIZE = 40  # Size of each board cell in pixels
BOARD_MARGIN = 40  # Margin around the board

# Global variables for the board display
pygame_initialized = False
board_display = None
display_last_move = None
display_title = ""

class GomokuGame:
    """Simple representation of a Gomoku game for simulation purposes."""
    
    def __init__(self):
        # Initialize board: channels [black, white, empty]
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        self.board[:, :, 2] = 1.0  # Mark all positions as empty
        self.current_player = True  # True for Black, False for White
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.board_history = []
        self.move_count = 0
        
    def make_move(self, row, col):
        """Make a move at the specified position."""
        # Convert from 1-indexed to 0-indexed
        r, c = row - 1, col - 1
        
        # Check if position is valid and empty
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.board[r, c, 2] != 1.0:  # If not empty
            return False
            
        # Place the stone
        channel = 0 if self.current_player else 1  # Black = 0, White = 1
        self.board[r, c, channel] = 1.0
        self.board[r, c, 2] = 0.0  # No longer empty
        
        # Record the move
        self.move_history.append((row, col))
        self.board_history.append(np.copy(self.board))
        self.move_count += 1
        
        # Check for win
        if self.check_win(r, c):
            self.game_over = True
            self.winner = self.current_player
            return True
            
        # Check for draw
        if np.sum(self.board[:, :, 2]) == 0:  # No empty spots
            self.game_over = True
            self.winner = None  # Draw
            return True
            
        # Switch player
        self.current_player = not self.current_player
        return True
        
    def check_win(self, row, col):
        """Check if the last move resulted in a win."""
        # Determine which player made the move
        is_black = self.current_player
        channel = 0 if is_black else 1
        
        # Directions: horizontal, vertical, diagonal (\), diagonal (/)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # Count in positive direction
            count_pos = 0
            for i in range(WIN_LENGTH):
                nx, ny = row + i * dx, col + i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny, channel] == 1.0:
                    count_pos += 1
                else:
                    break
                
            # Count in negative direction
            count_neg = 0
            for i in range(1, WIN_LENGTH):  # Start from 1 to avoid counting the current stone twice
                nx, ny = row - i * dx, col - i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny, channel] == 1.0:
                    count_neg += 1
                else:
                    break
                
            # Check if the total count meets the win condition
            if count_pos + count_neg >= WIN_LENGTH:
                return True
            
        return False
        
    def get_legal_moves(self):
        """Return list of legal moves as (row, col) tuples in 1-indexed form."""
        if self.game_over:
            return []
            
        legal_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c, 2] == 1.0:  # If empty
                    legal_moves.append((r+1, c+1))  # Convert to 1-indexed
                    
        return legal_moves
        
    def get_result(self):
        """Return game result: 1 for black win, -1 for white win, 0 for draw, None if ongoing."""
        if not self.game_over:
            return None
        if self.winner is None:
            return 0  # Draw
        return 1 if self.winner else -1  # 1 for black win, -1 for white win

def load_model(model_path, device='cuda'):
    """Load a GomokuNet model from a checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = GomokuNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def init_pygame_display():
    """Initialize the pygame display for board visualization."""
    global pygame_initialized, board_display
    
    if not pygame_initialized:
        pygame.init()
        pygame.display.set_caption("Gomoku AI Battle")
        board_width = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN
        board_height = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 50  # Extra space for title
        board_display = pygame.display.set_mode((board_width, board_height))
        pygame_initialized = True
    
    return board_display

def draw_board():
    """Draw the Gomoku board with stones."""
    global board_display, display_last_move, display_title
    
    if board_display is None:
        return
    
    # Background color
    board_display.fill((220, 180, 100))  # Light wood color
    
    # Draw the grid
    for i in range(BOARD_SIZE):
        # Vertical lines
        pygame.draw.line(
            board_display, 
            (0, 0, 0), 
            (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN),
            (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE),
            2 if i == 0 or i == BOARD_SIZE - 1 else 1
        )
        
        # Horizontal lines
        pygame.draw.line(
            board_display, 
            (0, 0, 0), 
            (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
            (BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
            2 if i == 0 or i == BOARD_SIZE - 1 else 1
        )
    
    # Draw star points (traditionally at 4-4, 10-10, etc for 15x15 board)
    star_points = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
    for x, y in star_points:
        pygame.draw.circle(
            board_display,
            (0, 0, 0),
            (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
            4
        )
    
    # Draw stones from the game board if available
    if hasattr(draw_board, 'current_board') and draw_board.current_board is not None:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if draw_board.current_board[i, j, 0] == 1.0:  # Black stone
                    pygame.draw.circle(
                        board_display,
                        (0, 0, 0),
                        (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                        CELL_SIZE // 2 - 2
                    )
                elif draw_board.current_board[i, j, 1] == 1.0:  # White stone
                    pygame.draw.circle(
                        board_display,
                        (255, 255, 255),
                        (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                        CELL_SIZE // 2 - 2
                    )
    
    # Highlight the last move
    if display_last_move is not None:
        row, col = display_last_move
        pygame.draw.circle(
            board_display,
            (255, 0, 0),  # Red highlight
            (BOARD_MARGIN + col * CELL_SIZE, BOARD_MARGIN + row * CELL_SIZE),
            5,
            2  # Width of the circle
        )
    
    # Display the title
    if display_title:
        font = pygame.font.SysFont(None, 28)
        title_surface = font.render(display_title, True, (0, 0, 0))
        board_display.blit(
            title_surface, 
            (BOARD_MARGIN, BOARD_MARGIN + BOARD_SIZE * CELL_SIZE + 20)
        )
    
    pygame.display.flip()

def update_display(board, last_move=None, title=""):
    """Update the board display with the current game state."""
    global display_last_move, display_title
    
    # Store the current board for drawing
    draw_board.current_board = board
    display_last_move = last_move
    display_title = title
    
    # Draw the updated board
    draw_board()
    
    # Process any events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

def play_game(model_black, model_white, temperature=0.8, max_moves=225, device='cuda', visualize=False, delay=0.5):
    """Simulate a full game between two models."""
    game = GomokuGame()
    models = {True: model_black, False: model_white}  # Map True->Black, False->White
    
    # For tracking policy confidence
    black_policy_confidence = []
    white_policy_confidence = []
    
    # Initialize visualization if requested
    if visualize:
        init_pygame_display()
        update_display(game.board, title="Game in Progress: Black's turn")
    
    while not game.game_over and game.move_count < max_moves:
        current_model = models[game.current_player]
        current_player_name = "Black" if game.current_player else "White"
        
        # Update display title
        if visualize:
            update_display(
                game.board, 
                title=f"Game in Progress: {current_player_name}'s turn (move {game.move_count+1})"
            )
        
        # Get model prediction
        try:
            top_move, probs, value = predict_move(
                current_model, 
                game.board, 
                temperature=temperature,
                move_count=game.move_count, 
                device=device
            )
            
            # Store highest probability for confidence tracking
            top_prob = np.max(probs)
            if game.current_player:  # Black
                black_policy_confidence.append(top_prob)
            else:  # White
                white_policy_confidence.append(top_prob)
                
            # Make the move
            success = game.make_move(top_move[0], top_move[1])
            
            # Update visualization
            if visualize:
                update_display(
                    game.board, 
                    last_move=(top_move[0]-1, top_move[1]-1),
                    title=f"Game in Progress: Move {game.move_count} - {current_player_name} played at {top_move}"
                )
                time.sleep(delay)  # Pause to make the game visible
            
            if not success:
                # Model tried an illegal move
                print(f"Model made illegal move: {top_move}")
                # Pick a random legal move instead
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    random_move = legal_moves[np.random.randint(len(legal_moves))]
                    game.make_move(random_move[0], random_move[1])
                    
                    # Update visualization for the random move
                    if visualize:
                        update_display(
                            game.board, 
                            last_move=(random_move[0]-1, random_move[1]-1),
                            title=f"Game in Progress: Move {game.move_count} - {current_player_name} played RANDOM at {random_move}"
                        )
                        time.sleep(delay)
                else:
                    # No legal moves, game is a draw
                    game.game_over = True
                    game.winner = None
                    
        except Exception as e:
            print(f"Error during model prediction: {str(e)}")
            # Handle failure by picking a random move
            legal_moves = game.get_legal_moves()
            if legal_moves:
                random_move = legal_moves[np.random.randint(len(legal_moves))]
                game.make_move(random_move[0], random_move[1])
                
                # Update visualization for the random move
                if visualize:
                    update_display(
                        game.board, 
                        last_move=(random_move[0]-1, random_move[1]-1),
                        title=f"Game in Progress: Move {game.move_count} - {current_player_name} played RANDOM at {random_move}"
                    )
                    time.sleep(delay)
            else:
                game.game_over = True
                game.winner = None
    
    # Show final board and result if visualizing
    if visualize:
        result_text = "Draw"
        if game.winner is not None:
            result_text = "Black wins" if game.winner else "White wins"
        update_display(
            game.board, 
            title=f"Game over: {result_text} in {game.move_count} moves"
        )
        time.sleep(2.0)  # Show final position for 2 seconds
    
    # Return the game result and additional stats
    return {
        'result': game.get_result(),
        'moves': game.move_count,
        'move_history': game.move_history,
        'board_history': game.board_history,
        'black_confidence': np.mean(black_policy_confidence) if black_policy_confidence else 0,
        'white_confidence': np.mean(white_policy_confidence) if white_policy_confidence else 0
    }

def run_battle(model_a_path, model_b_path, num_games=100, temperature=0.8, swap=True, 
               device='cuda', save_results=True, output_dir='battle_results', visualize=False, delay=0.5):
    """
    Run a battle between two models.
    
    Args:
        model_a_path: Path to first model checkpoint
        model_b_path: Path to second model checkpoint
        num_games: Number of games to play (half for each color if swap=True)
        temperature: Temperature for softmax sampling
        swap: Whether to swap colors halfway through
        device: 'cuda' or 'cpu'
        save_results: Whether to save detailed results
        output_dir: Directory to save results to
        visualize: Whether to visualize the games
        delay: Delay between moves when visualizing (seconds)
    
    Returns:
        Dictionary with battle results
    """
    # Create output directory if needed
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print(f"Loading models...")
    model_a = load_model(model_a_path, device)
    model_b = load_model(model_b_path, device)
    
    # Get model names from paths
    model_a_name = os.path.basename(model_a_path).replace('.pth', '')
    model_b_name = os.path.basename(model_b_path).replace('.pth', '')
    
    # Initialize counters
    results = {
        'a_wins_as_black': 0,
        'b_wins_as_black': 0,
        'a_wins_as_white': 0, 
        'b_wins_as_white': 0,
        'draws': 0,
        'avg_game_length': 0,
        'a_avg_confidence': 0,
        'b_avg_confidence': 0,
        'games': []
    }
    
    total_moves = 0
    total_a_confidence_black = []
    total_b_confidence_black = []
    total_a_confidence_white = []
    total_b_confidence_white = []
    
    # Determine how many games each model plays as black
    if swap:
        games_per_side = num_games // 2
    else:
        games_per_side = num_games
    
    # Play games with Model A as Black, Model B as White
    print(f"\nPlaying {games_per_side} games with {model_a_name} as Black vs {model_b_name} as White")
    for i in tqdm(range(games_per_side)):
        game_result = play_game(model_a, model_b, temperature, device=device, visualize=visualize, delay=delay)
        total_moves += game_result['moves']
        
        # Store confidence values
        total_a_confidence_black.extend([game_result['black_confidence']])
        total_b_confidence_white.extend([game_result['white_confidence']])
        
        # Record the outcome
        if game_result['result'] == 1:  # Black (A) wins
            results['a_wins_as_black'] += 1
        elif game_result['result'] == -1:  # White (B) wins
            results['b_wins_as_white'] += 1
        else:  # Draw
            results['draws'] += 1
            
        # Store detailed game info
        results['games'].append({
            'black': model_a_name,
            'white': model_b_name,
            'result': game_result['result'],
            'moves': game_result['moves'],
            'black_confidence': game_result['black_confidence'],
            'white_confidence': game_result['white_confidence'],
            'move_history': str(game_result['move_history'])  # Add move history to the CSV
        })
    
    # If swapping colors, play games with Model B as Black, Model A as White
    if swap:
        print(f"\nPlaying {games_per_side} games with {model_b_name} as Black vs {model_a_name} as White")
        for i in tqdm(range(games_per_side)):
            game_result = play_game(model_b, model_a, temperature, device=device, visualize=visualize, delay=delay)
            total_moves += game_result['moves']
            
            # Store confidence values
            total_b_confidence_black.extend([game_result['black_confidence']])
            total_a_confidence_white.extend([game_result['white_confidence']])
            
            # Record the outcome
            if game_result['result'] == 1:  # Black (B) wins
                results['b_wins_as_black'] += 1
            elif game_result['result'] == -1:  # White (A) wins
                results['a_wins_as_white'] += 1
            else:  # Draw
                results['draws'] += 1
                
            # Store detailed game info
            results['games'].append({
                'black': model_b_name,
                'white': model_a_name,
                'result': game_result['result'],
                'moves': game_result['moves'],
                'black_confidence': game_result['black_confidence'],
                'white_confidence': game_result['white_confidence'],
                'move_history': str(game_result['move_history'])  # Add move history to the CSV
            })
    else:
        # This block correctly reflects the logic from lines 409-411
        # but its previous position caused the error.
        # No explicit action needed here if the only purpose was setting games_per_side,
        # as that was handled earlier when swap was determined to be False.
        # If there was other intended logic for the non-swap case that should happen
        # *after* the first loop, it would go here.
        # For now, just having the correctly indented else is sufficient.
        pass
    
    # Clean up pygame if it was initialized
    if visualize and pygame_initialized:
        pygame.quit()
    
    # Calculate overall statistics
    results['avg_game_length'] = total_moves / num_games
    
    # Calculate average confidence for each model
    if total_a_confidence_black and total_a_confidence_white:
        results['a_avg_confidence'] = np.mean(total_a_confidence_black + total_a_confidence_white)
    
    if total_b_confidence_black and total_b_confidence_white:
        results['b_avg_confidence'] = np.mean(total_b_confidence_black + total_b_confidence_white)
    
    # Calculate total wins for each model
    results['a_total_wins'] = results['a_wins_as_black'] + results['a_wins_as_white']
    results['b_total_wins'] = results['b_wins_as_black'] + results['b_wins_as_white']
    
    # Calculate win rates
    results['a_win_rate'] = results['a_total_wins'] / num_games
    results['b_win_rate'] = results['b_total_wins'] / num_games
    results['draw_rate'] = results['draws'] / num_games
    
    # Elo estimate (very approximate)
    if results['a_win_rate'] > 0 and results['b_win_rate'] > 0:
        elo_diff = 400 * np.log10(results['a_win_rate'] / results['b_win_rate'])
        results['estimated_elo_diff'] = elo_diff
    else:
        results['estimated_elo_diff'] = float('inf') if results['a_win_rate'] > 0 else float('-inf')
    
    # Save results to CSV if requested
    if save_results:
        # Create detailed games DataFrame
        games_df = pd.DataFrame(results['games'])
        games_df.to_csv(f"{output_dir}/games_{model_a_name}_vs_{model_b_name}.csv", index=False)
        
        # Create summary DataFrame with a single row
        summary = {k: [v] for k, v in results.items() if k != 'games'}
        summary['model_a'] = [model_a_name]
        summary['model_b'] = [model_b_name]
        summary['timestamp'] = [time.strftime("%Y-%m-%d %H:%M:%S")]
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/summary_{model_a_name}_vs_{model_b_name}.csv", index=False)
        
        # Create visualizations
        create_battle_visualizations(results, model_a_name, model_b_name, output_dir)
        
    # Print summary
    print_battle_summary(results, model_a_name, model_b_name)
    
    return results

def print_battle_summary(results, model_a_name, model_b_name):
    """Print a nice formatted summary of battle results."""
    print("\n" + "="*50)
    print(f"BATTLE RESULTS: {model_a_name} vs {model_b_name}")
    print("="*50)
    
    # Create a table for the results
    table = PrettyTable()
    table.field_names = ["Metric", model_a_name, model_b_name]
    
    # Add general stats
    table.add_row(["Total Games", results['a_total_wins'] + results['b_total_wins'] + results['draws'], 
                  results['a_total_wins'] + results['b_total_wins'] + results['draws']])
    table.add_row(["Wins", results['a_total_wins'], results['b_total_wins']])
    table.add_row(["Win Rate", f"{results['a_win_rate']:.2%}", f"{results['b_win_rate']:.2%}"])
    table.add_row(["Wins as Black", results['a_wins_as_black'], results['b_wins_as_black']])
    table.add_row(["Wins as White", results['a_wins_as_white'], results['b_wins_as_white']])
    table.add_row(["Avg Confidence", f"{results['a_avg_confidence']:.4f}", f"{results['b_avg_confidence']:.4f}"])
    
    # Add shared stats
    table.add_row(["Draws", results['draws'], results['draws']])
    table.add_row(["Avg Game Length", f"{results['avg_game_length']:.1f} moves", f"{results['avg_game_length']:.1f} moves"])
    
    # Add Elo difference if calculated
    if 'estimated_elo_diff' in results:
        if np.isfinite(results['estimated_elo_diff']):
            elo_text = f"{abs(results['estimated_elo_diff']):.1f} Elo"
            if results['estimated_elo_diff'] > 0:
                table.add_row(["Elo Advantage", elo_text, ""])
            else:
                table.add_row(["Elo Advantage", "", elo_text])
    
    print(table)
    print("\n" + "="*50)
    
    # Print conclusion
    if results['a_total_wins'] > results['b_total_wins']:
        winner = model_a_name
        win_margin = results['a_total_wins'] - results['b_total_wins']
        win_percentage = results['a_win_rate']
    elif results['b_total_wins'] > results['a_total_wins']:
        winner = model_b_name
        win_margin = results['b_total_wins'] - results['a_total_wins']
        win_percentage = results['b_win_rate']
    else:
        print("The battle ended in a draw! Both models performed equally.")
        return
    
    print(f"WINNER: {winner} won by {win_margin} games ({win_percentage:.2%} win rate)")
    print("="*50 + "\n")

def create_battle_visualizations(results, model_a_name, model_b_name, output_dir):
    """Create visualizations of battle results."""
    # 1. Wins by color
    plt.figure(figsize=(12, 8))
    
    # Data for the grouped bar chart
    models = [model_a_name, model_b_name]
    wins_as_black = [results['a_wins_as_black'], results['b_wins_as_black']]
    wins_as_white = [results['a_wins_as_white'], results['b_wins_as_white']]
    
    # Set width of bars
    barWidth = 0.3
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    
    # Create grouped bars
    plt.bar(r1, wins_as_black, width=barWidth, label='Wins as Black', color='black')
    plt.bar(r2, wins_as_white, width=barWidth, label='Wins as White', color='lightgray', edgecolor='black')
    
    # Add data labels above bars
    for i, v in enumerate(wins_as_black):
        plt.text(i, v + 0.1, str(v), ha='center')
        
    for i, v in enumerate(wins_as_white):
        plt.text(i + barWidth, v + 0.1, str(v), ha='center')
    
    # Add xticks in the middle of the group bars
    plt.xticks([r + barWidth/2 for r in range(len(models))], models)
    plt.xlabel('Model')
    plt.ylabel('Number of Wins')
    plt.title('Wins by Color')
    plt.legend()
    
    # Save the figure
    plt.savefig(f"{output_dir}/wins_by_color_{model_a_name}_vs_{model_b_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pie chart of overall results
    plt.figure(figsize=(10, 10))
    
    # Data for pie chart
    labels = [f"{model_a_name} Wins", f"{model_b_name} Wins", "Draws"]
    sizes = [results['a_total_wins'], results['b_total_wins'], results['draws']]
    colors = ['#3498db', '#e74c3c', '#95a5a6']  # Blue, Red, Gray
    
    # Create pie chart
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Overall Results')
    
    # Save the figure
    plt.savefig(f"{output_dir}/overall_results_{model_a_name}_vs_{model_b_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Game length histogram (if enough games)
    if len(results['games']) >= 10:
        plt.figure(figsize=(12, 8))
        
        # Extract game lengths
        game_lengths = [game['moves'] for game in results['games']]
        
        # Create histogram
        plt.hist(game_lengths, bins=20, alpha=0.7, color='#2ecc71')
        plt.axvline(results['avg_game_length'], color='red', linestyle='dashed', linewidth=1)
        plt.text(results['avg_game_length'] + 1, plt.ylim()[1] * 0.9, 
                 f'Average: {results["avg_game_length"]:.1f} moves', 
                 color='red')
        
        plt.xlabel('Game Length (moves)')
        plt.ylabel('Number of Games')
        plt.title('Distribution of Game Lengths')
        
        # Save the figure
        plt.savefig(f"{output_dir}/game_lengths_{model_a_name}_vs_{model_b_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

class ModelBattleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku Model Battle")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selection
        ttk.Label(main_frame, text="Model A:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_a_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.model_a_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=lambda: self.browse_model("a")).grid(row=0, column=2, pady=5)
        
        ttk.Label(main_frame, text="Model B:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_b_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.model_b_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=lambda: self.browse_model("b")).grid(row=1, column=2, pady=5)
        
        # Battle settings
        settings_frame = ttk.LabelFrame(main_frame, text="Battle Settings")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(settings_frame, text="Number of Games:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.num_games = tk.IntVar(value=20)
        ttk.Spinbox(settings_frame, from_=2, to=1000, textvariable=self.num_games, width=10).grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(settings_frame, text="Temperature:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
        self.temperature = tk.DoubleVar(value=0.8)
        ttk.Spinbox(settings_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.temperature, width=10).grid(row=0, column=3, sticky=tk.W, pady=5, padx=5)
        
        self.swap_colors = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Swap Colors (each model plays as both black and white)", variable=self.swap_colors).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5, padx=5)
        
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        ttk.Checkbutton(settings_frame, text="Use GPU (CUDA)", variable=self.use_gpu).grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=5, padx=5)
        
        # Visualization settings
        self.visualize_games = tk.BooleanVar(value=True)  # Default to True
        ttk.Checkbutton(settings_frame, text="Visualize Games", variable=self.visualize_games).grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(settings_frame, text="Move Delay (sec):").grid(row=3, column=2, sticky=tk.W, pady=5, padx=5)
        self.move_delay = tk.DoubleVar(value=0.5)
        ttk.Spinbox(settings_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.move_delay, width=10).grid(row=3, column=3, sticky=tk.W, pady=5, padx=5)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_dir = tk.StringVar(value="battle_results")
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_dir).grid(row=3, column=2, pady=5)
        
        # Start button
        ttk.Button(main_frame, text="Start Battle", command=self.start_battle).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Select model files and configure settings")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=580, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=5)
    
    def browse_model(self, model_id):
        filename = filedialog.askopenfilename(
            title=f"Select Model {model_id.upper()}",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            if model_id == "a":
                self.model_a_path.set(filename)
            else:
                self.model_b_path.set(filename)
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def start_battle(self):
        # Validate inputs
        if not self.model_a_path.get():
            self.status_var.set("Error: Model A not selected")
            return
        
        if not self.model_b_path.get():
            self.status_var.set("Error: Model B not selected")
            return
        
        # Get parameters
        model_a_path = self.model_a_path.get()
        model_b_path = self.model_b_path.get()
        num_games = self.num_games.get()
        temperature = self.temperature.get()
        swap = self.swap_colors.get()
        device = "cuda" if self.use_gpu.get() and torch.cuda.is_available() else "cpu"
        output_dir = self.output_dir.get()
        visualize = self.visualize_games.get()
        delay = self.move_delay.get()
        
        # Show progress
        self.status_var.set("Starting battle... This may take a while")
        self.progress.start()
        self.root.update()
        
        # Run battle in a separate thread to keep UI responsive
        def run():
            try:
                # Don't use threading with pygame - it can cause issues
                # Instead, we'll let pygame handle its own window and event loop
                results = run_battle(
                    model_a_path=model_a_path,
                    model_b_path=model_b_path,
                    num_games=num_games,
                    temperature=temperature,
                    swap=swap,
                    device=device,
                    output_dir=output_dir,
                    visualize=visualize,
                    delay=delay
                )
                
                # When complete, update UI
                self.root.after(0, lambda: self.battle_complete(results))
            except Exception as e:
                import traceback
                self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
                self.root.after(0, self.progress.stop)
                print(traceback.format_exc())
        
        battle_thread = threading.Thread(target=run)
        battle_thread.daemon = True
        battle_thread.start()
    
    def battle_complete(self, results):
        self.progress.stop()
        model_a_name = os.path.basename(self.model_a_path.get()).replace('.pth', '')
        model_b_name = os.path.basename(self.model_b_path.get()).replace('.pth', '')
        
        if results['a_total_wins'] > results['b_total_wins']:
            winner = model_a_name
            win_rate = results['a_win_rate']
        elif results['b_total_wins'] > results['a_total_wins']:
            winner = model_b_name
            win_rate = results['b_win_rate']
        else:
            winner = "Draw"
            win_rate = 0.5
        
        self.status_var.set(f"Battle complete! Winner: {winner} ({win_rate:.1%}) - Results saved to {self.output_dir.get()}")
        
        # Optionally open the output directory
        import platform
        if platform.system() == 'Windows':
            os.startfile(self.output_dir.get())
        elif platform.system() == 'Darwin':  # macOS
            import subprocess
            subprocess.call(['open', self.output_dir.get()])
        else:  # Linux
            import subprocess
            subprocess.call(['xdg-open', self.output_dir.get()])

if __name__ == "__main__":
    # Check if running in GUI or command-line mode
    if len(sys.argv) > 1:
        # Command-line mode
        parser = argparse.ArgumentParser(description="Run a battle between two Gomoku models")
        parser.add_argument("--model_a", type=str, required=True, help="Path to first model checkpoint")
        parser.add_argument("--model_b", type=str, required=True, help="Path to second model checkpoint")
        parser.add_argument("--num_games", type=int, default=100, help="Number of games to play")
        parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for move sampling")
        parser.add_argument("--no_swap", action="store_true", help="Don't swap colors halfway through")
        parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
        parser.add_argument("--output_dir", type=str, default="battle_results", help="Directory to save results")
        parser.add_argument("--visualize", action="store_true", help="Visualize the games as they are played")
        parser.add_argument("--delay", type=float, default=0.5, help="Delay between moves when visualizing (seconds)")
        
        args = parser.parse_args()
        
        # Run the battle
        results = run_battle(
            model_a_path=args.model_a,
            model_b_path=args.model_b,
            num_games=args.num_games,
            temperature=args.temperature,
            swap=not args.no_swap,
            device=args.device,
            output_dir=args.output_dir,
            visualize=args.visualize,
            delay=args.delay
        )
    else:
        # GUI mode
        root = tk.Tk()
        app = ModelBattleGUI(root)
        root.mainloop() 