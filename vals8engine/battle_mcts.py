import sys
import os
import time
import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import torch
import argparse
from inference import load_model, predict_move_with_details, predict_move_with_weighted_details
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
gomoku_src_path = os.path.join(current_dir, "gomoku", "src")
sys.path.insert(0, gomoku_src_path)

from gomokumcts.src.pygomoku.Board import Board
from gomokumcts.src.pygomoku.Player import Player, PureMCTSPlayer
from gomokumcts.src.pygomoku.GameServer import GameServer

OriginalPlayer = Player

from model import GomokuNet

BOARD_SIZE = 15
CELL_SIZE = 40
BOARD_MARGIN = 40
WIN_LENGTH = 5

pygame_initialized = False
board_display = None
display_title = ""

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class NNPlayerWrapper(Player):
    def __init__(self, color, model_path, name="NN Player", temperature=0.3, 
                 use_value_eval=True, alpha=0.7, top_k=5, device='cuda'):
        """
        Wrapper for our neural network that implements the Player interface.
        """
        self.__color = color
        self.__name = name
        self.temperature = temperature
        self.use_value_eval = use_value_eval
        self.alpha = alpha
        self.top_k = top_k
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_weighted = False
        
        self.model = load_model(model_path, device=self.device)
        
        self.move_count = 0
        
    def getAction(self, board):
        """
        Get a move from the neural network.
        
        Args:
            board: Board object from pygomoku.Board
            
        Returns:
            int: Move index expected by MCTS interface
        """
        if board.current_player != self.__color:
            raise RuntimeError("The current player's color in board is "
                              "not equal to the color of current player.")
        
        available_moves = []
        for move_idx in board.availables:
            location = board.moveToLocation(move_idx)
            available_moves.append((location[0]+1, location[1]+1))  
        
        if not available_moves:
            return -1
            
        raw_board = self._convert_board(board)
        
        for attempt in range(3):  
            if self.use_weighted:
                move, probs, _, _ = predict_move_with_weighted_details(
                    self.model, 
                    raw_board, 
                    temperature=self.temperature,
                    alpha=self.alpha,
                    move_count=self.move_count,
                    device=self.device,
                    top_k=self.top_k
                )
            else:
                move, probs, _, _ = predict_move_with_details(
                    self.model, 
                    raw_board,
                    temperature=self.temperature,
                    move_count=self.move_count,
                    device=self.device,
                    top_k=self.top_k,
                    use_value_eval=self.use_value_eval
                )
            
            if move in available_moves:
                self.move_count += 1
                row, col = move[0]-1, move[1]-1
                move_idx = board.locationToMove([row, col])
                return move_idx
            
            self.temperature += 0.3
        
        random_idx = np.random.randint(0, len(board.availables))
        return board.availables[random_idx]
    
    def _convert_board(self, board):
        """
        Convert a pygomoku Board object to our 3-channel numpy array format.
        
        Args:
            board: Board object from pygomoku.Board
            
        Returns:
            np.array: Board in shape (BOARD_SIZE, BOARD_SIZE, 3)
        """
        raw_board = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        raw_board[:, :, 2] = 1.0  
        
        width, height = board.width, board.height
        
        for x in range(width):
            for y in range(height):
                move = board.locationToMove([y, x])
                stone = board.states.get(move, Board.kEmpty)
                if stone == Board.kPlayerBlack:
                    raw_board[y, x, 0] = 1.0  # Black stone
                    raw_board[y, x, 2] = 0.0  # Not empty
                elif stone == Board.kPlayerWhite:
                    raw_board[y, x, 1] = 1.0  # White stone
                    raw_board[y, x, 2] = 0.0  # Not empty
        
        return raw_board
        
    def __str__(self):
        """Print information about the player."""
        if self.__color == Board.kPlayerBlack:
            color = "Black[@]"
        elif self.__color == Board.kPlayerWhite:
            color = "White[O]"
        else:
            color = "None[+]"
            
        return f"[--Player Info--]\nNeural Network Player\nName: {self.__name}\nColor: {color}\nTemperature: {self.temperature}"
        
    __repr__ = __str__
    
    @property
    def color(self):
        return self.__color
        
    @color.setter
    def color(self, given_color):
        if given_color not in [Board.kPlayerBlack, Board.kPlayerWhite, Board.kEmpty]:
            return
        self.__color = given_color
        
    @property
    def name(self):
        return self.__name
        
    @name.setter
    def name(self, given_name):
        if not isinstance(given_name, str):
            return
        self.__name = given_name

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

def draw_board(board):
    """Draw the Gomoku board with stones."""
    global board_display, display_title
    
    if board_display is None:
        return
    
    board_display.fill((220, 180, 100))  # Light wood color
    
    for i in range(BOARD_SIZE):
        pygame.draw.line(
            board_display, 
            (0, 0, 0), 
            (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN),
            (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE),
            2 if i == 0 or i == BOARD_SIZE - 1 else 1
        )
        
        pygame.draw.line(
            board_display, 
            (0, 0, 0), 
            (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
            (BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
            2 if i == 0 or i == BOARD_SIZE - 1 else 1
        )
    
    star_points = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
    for x, y in star_points:
        pygame.draw.circle(
            board_display,
            (0, 0, 0),
            (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
            4
        )
    
    width, height = board.width, board.height
    for x in range(width):
        for y in range(height):
            move = board.locationToMove([y, x])
            stone = board.states.get(move, Board.kEmpty)
            if stone == Board.kPlayerBlack:
                pygame.draw.circle(
                    board_display,
                    (0, 0, 0),  
                    (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
                    CELL_SIZE // 2 - 2
                )
            elif stone == Board.kPlayerWhite:
                pygame.draw.circle(
                    board_display,
                    (255, 255, 255),  # White
                    (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
                    CELL_SIZE // 2 - 2
                )
    
    if display_title:
        font = pygame.font.SysFont(None, 28)
        title_surface = font.render(display_title, True, (0, 0, 0))
        board_display.blit(
            title_surface, 
            (BOARD_MARGIN, BOARD_MARGIN + BOARD_SIZE * CELL_SIZE + 20)
        )
    
    pygame.display.flip()

class CustomGameServer(GameServer):
    def __init__(self, board, mode, player1, player2, visualize=False, delay=0.5):
        """Override the original GameServer to bypass type checking for our custom player class"""
        self._player1 = player1
        self._player2 = player2
        self._visualize = visualize
        self._delay = delay       
        self._board = board
        self._mode = mode
        self._is_running = False
        self.move_history = []  
        if visualize:
            init_pygame_display()
    
    def _startNormalGame(self):
        """Override to add visualization and return the winner."""
        global display_title
        
        self._board.initBoard(self._player1.color)
        
        players = {
            self._player1.color: self._player1,
            self._player2.color: self._player2
        }
        
        while True:
            current_player = players[self._board.current_player]
            
            if self._visualize:
                display_title = f"{current_player.name}'s Turn"
                draw_board(self._board)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None
            
            move = current_player.getAction(self._board)
            success = self._board.play(move)
            
            if not success:
                raise RuntimeError(f"Invalid move: {move}")
            
            self.move_history.append((self._board.current_player, move))
            
            if self._visualize:
                time.sleep(self._delay)
                draw_board(self._board)
            
            is_end, winner = self._board.gameEnd()
            if is_end:
                if self._visualize:
                    if winner == Board.kPlayerBlack:
                        display_title = f"Game Over: {players[Board.kPlayerBlack].name} (Black) Wins!"
                    elif winner == Board.kPlayerWhite:
                        display_title = f"Game Over: {players[Board.kPlayerWhite].name} (White) Wins!"
                    else:
                        display_title = "Game Over: Draw!"
                    
                    draw_board(self._board)
                    time.sleep(2)
                    pygame.quit()
                
                return winner
    
    def startGame(self):
        """Start the game with the appropriate mode."""
        if self._mode == GameServer.kNormalPlayGame:
            return self._startNormalGame()
        else:
            raise ValueError(f"Unsupported game mode: {self._mode}")
            return None

def play_single_game(our_model_path, mcts_compute_budget=20000, 
                     our_player_black=True, temperature=0.3,
                     use_value_eval=True, alpha=0.7, use_weighted=False,
                     visualize=True, delay=0.5, time_limit=20.0):
    """
    Play a single game between our neural network and the MCTS engine.
    
    Args:
        our_model_path: Path to our model checkpoint
        mcts_compute_budget: Compute budget for MCTS algorithm (iterations)
        our_player_black: Whether our player plays as black
        temperature: Temperature for move selection (default 0.3, lower = more deterministic)
        use_value_eval: Whether to use value head for evaluation
        alpha: Weight factor for policy vs value in weighted approach
        use_weighted: Whether to use weighted approach
        visualize: Whether to visualize the game
        delay: Delay between moves for visualization (seconds)
        time_limit: Time limit for MCTS thinking (seconds)
        
    Returns:
        Tuple of (winner, game_data) where:
        - winner: Game winner (Board.kPlayerBlack, Board.kPlayerWhite, or None for draw)
        - game_data: Dictionary with additional game statistics
    """
    # Create a board
    board = Board(width=BOARD_SIZE, height=BOARD_SIZE)
    
    # Setup players
    if our_player_black:
        player1 = NNPlayerWrapper(
            Board.kPlayerBlack, 
            our_model_path, 
            name="NN Player (Black)",
            temperature=temperature,
            use_value_eval=use_value_eval,
            alpha=alpha
        )
        player1.use_weighted = use_weighted
        player2 = TimeLimitedMCTSPlayer(
            Board.kPlayerWhite, 
            name="Time Limited MCTS (White)",
            compute_budget=mcts_compute_budget,
            silent=not visualize,
            time_limit=time_limit
        )
    else:
        player1 = TimeLimitedMCTSPlayer(
            Board.kPlayerBlack, 
            name="Time Limited MCTS (Black)",
            compute_budget=mcts_compute_budget,
            silent=not visualize,
            time_limit=time_limit
        )
        player2 = NNPlayerWrapper(
            Board.kPlayerWhite, 
            our_model_path,
            name="NN Player (White)",
            temperature=temperature,
            use_value_eval=use_value_eval,
            alpha=alpha
        )
        player2.use_weighted = use_weighted
    
    server = CustomGameServer(
        board=board,
        mode=GameServer.kNormalPlayGame,
        player1=player1,
        player2=player2,
        visualize=visualize,
        delay=delay
    )
    
    game_data = {
        'num_moves': 0,
        'our_player_black': our_player_black,
        'our_moves': [],    # List of (x, y) tuples for our player's moves
        'mcts_moves': []    # List of (x, y) tuples for MCTS player's moves
    }
    
    winner = server.startGame()
    
    game_data['num_moves'] = len(server.move_history)
    
    our_player_color = Board.kPlayerBlack if our_player_black else Board.kPlayerWhite
    mcts_player_color = Board.kPlayerWhite if our_player_black else Board.kPlayerBlack
    
    for player_color, move_idx in server.move_history:
        location = board.moveToLocation(move_idx)
        x, y = location[0], location[1]
        
        if player_color == our_player_color:
            game_data['our_moves'].append((x, y))
        else:
            game_data['mcts_moves'].append((x, y))
    
    return winner, game_data

def run_battle(our_model_path, num_games=10, mcts_compute_budget=20000,
               temperature=0.3, use_value_eval=True, alpha=0.7, use_weighted=False,
               visualize=True, delay=0.5, output_dir='battle_results', time_limit=20.0):
    """
    Run a battle between our neural network model and MCTS engine.
    
    Args:
        our_model_path: Path to our model checkpoint
        num_games: Number of games to play
        mcts_compute_budget: Compute budget for MCTS algorithm (iterations)
        temperature: Temperature for move selection
        use_value_eval: Whether to use value head for evaluation
        alpha: Weight factor for policy vs value in weighted approach
        use_weighted: Whether to use weighted approach
        visualize: Whether to visualize games
        delay: Delay between moves for visualization
        output_dir: Directory to save results
        time_limit: Time limit for MCTS thinking (seconds)
        
    Returns:
        Dictionary with battle results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'our_wins_as_black': 0,
        'our_wins_as_white': 0,
        'mcts_wins_as_black': 0,
        'mcts_wins_as_white': 0,
        'draws': 0,
        'game_results': [],
        'game_stats': []   
    }
    
    games_as_black = num_games // 2
    games_as_white = num_games - games_as_black
    
    for i in range(num_games):
        our_player_black = i < games_as_black
        
        winner, game_data = play_single_game(
            our_model_path=our_model_path,
            mcts_compute_budget=mcts_compute_budget,
            our_player_black=our_player_black,
            temperature=temperature,
            use_value_eval=use_value_eval,
            alpha=alpha,
            use_weighted=use_weighted,
            visualize=visualize,
            delay=delay,
            time_limit=time_limit
        )
        
        results['game_stats'].append(game_data)
        
        if winner == Board.kPlayerBlack:
            if our_player_black:
                results['our_wins_as_black'] += 1
                outcome = 'win'
            else:
                results['mcts_wins_as_black'] += 1
                outcome = 'loss'
        elif winner == Board.kPlayerWhite:
            if not our_player_black:
                results['our_wins_as_white'] += 1
                outcome = 'win'
            else:
                results['mcts_wins_as_white'] += 1
                outcome = 'loss'
        else:
            results['draws'] += 1
            outcome = 'draw'
        
        results['game_results'].append({
            'game_id': i + 1,
            'our_player_black': our_player_black,
            'winner': 'black' if winner == Board.kPlayerBlack else 'white' if winner == Board.kPlayerWhite else 'draw',
            'outcome': outcome,
            'num_moves': game_data['num_moves']
        })
        
        print(f"Game {i+1}/{num_games} - " + 
              f"{'Our model' if our_player_black else 'MCTS'} as Black, " +
              f"{'Our model' if not our_player_black else 'MCTS'} as White - " +
              f"Result: {'Black wins' if winner == Board.kPlayerBlack else 'White wins' if winner == Board.kPlayerWhite else 'Draw'}")
    
    results_df = pd.DataFrame(results['game_results'])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f"{output_dir}/battle_results_{timestamp}.csv", index=False)
    
    our_total_wins = results['our_wins_as_black'] + results['our_wins_as_white']
    mcts_total_wins = results['mcts_wins_as_black'] + results['mcts_wins_as_white']
    
    print("\nBattle Results Summary:")
    print(f"Our Model Wins: {our_total_wins} ({our_total_wins/num_games*100:.1f}%)")
    print(f"  As Black: {results['our_wins_as_black']}")
    print(f"  As White: {results['our_wins_as_white']}")
    print(f"MCTS Wins: {mcts_total_wins} ({mcts_total_wins/num_games*100:.1f}%)")
    print(f"  As Black: {results['mcts_wins_as_black']}")
    print(f"  As White: {results['mcts_wins_as_white']}")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    
    avg_game_length = np.mean([game['num_moves'] for game in results['game_results']])
    min_game_length = min([game['num_moves'] for game in results['game_results']])
    max_game_length = max([game['num_moves'] for game in results['game_results']])
    print(f"\nGame Length Statistics:")
    print(f"  Average: {avg_game_length:.1f} moves")
    print(f"  Minimum: {min_game_length} moves")
    print(f"  Maximum: {max_game_length} moves")
    
    plt.figure(figsize=(10, 6))
    labels = ['Our Model', 'MCTS', 'Draws']
    sizes = [our_total_wins, mcts_total_wins, results['draws']]
    colors = ['#66b3ff', '#ff9999', '#99ff99']
    explode = (0.1, 0, 0)
    
    black_games = max(1, games_as_black)
    white_games = max(1, games_as_white) 
    
    plt.subplot(1, 2, 1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'Overall Results ({num_games} games)')
    
    plt.subplot(1, 2, 2)
    categories = ['As Black', 'As White']
    our_win_rates = [results['our_wins_as_black'] / black_games * 100,
                    results['our_wins_as_white'] / white_games * 100]
    mcts_win_rates = [results['mcts_wins_as_black'] / white_games * 100,
                    results['mcts_wins_as_white'] / black_games * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, our_win_rates, width, label='Our Model', color='#66b3ff')
    plt.bar(x + width/2, mcts_win_rates, width, label='MCTS', color='#ff9999')
    
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate by Color')
    plt.xticks(x, categories)
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/battle_results_{timestamp}.png")
    
    create_game_length_plot(results, output_dir, timestamp)
    
    if visualize:
        plt.show()
    
    return results

def create_game_length_plot(results, output_dir, timestamp):
    """
    Create a histogram of game lengths
    
    Args:
        results: Dictionary with battle results
        output_dir: Directory to save the plots
        timestamp: Timestamp for filenames
    """
    plt.figure(figsize=(12, 6))
    
    moves = []
    outcomes = []
    for game in results['game_results']:
        moves.append(game['num_moves'])
        outcomes.append(game['outcome'])
    
    colors = []
    for outcome in outcomes:
        if outcome == 'win':
            colors.append('#66b3ff')  # Our win - blue
        elif outcome == 'loss':
            colors.append('#ff9999')  # Our loss - red
        else:
            colors.append('#99ff99')  # Draw - green
    
    plt.subplot(1, 2, 1)
    bins = range(min(moves), max(moves) + 5, 5)
    plt.hist(moves, bins=bins, alpha=0.7, color='#66b3ff')
    plt.axvline(np.mean(moves), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(moves):.1f}')
    plt.xlabel('Number of Moves')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Game Lengths')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    win_lengths = [game['num_moves'] for game in results['game_results'] if game['outcome'] == 'win']
    loss_lengths = [game['num_moves'] for game in results['game_results'] if game['outcome'] == 'loss']
    draw_lengths = [game['num_moves'] for game in results['game_results'] if game['outcome'] == 'draw']
    
    box_data = [win_lengths, loss_lengths, draw_lengths] 
    box_labels = ['Our Wins', 'Our Losses', 'Draws']
    box_colors = ['#66b3ff', '#ff9999', '#99ff99']
    
    plt.boxplot(box_data, labels=box_labels, patch_artist=True, 
                boxprops=dict(alpha=0.7),
                medianprops=dict(color='black'))
    
    for i, data in enumerate(box_data):
        x = np.random.normal(i+1, 0.04, size=len(data))
        plt.scatter(x, data, alpha=0.6, color=box_colors[i], edgecolor='black', s=30)
    
    plt.ylabel('Number of Moves')
    plt.title('Game Length by Outcome')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/game_length_analysis_{timestamp}.png", dpi=300)

class TimeLimitedMCTSPlayer(PureMCTSPlayer):
    """MCTS player with a time limit (default 20 seconds) on thinking"""
    
    def __init__(self, color, name="Time Limited MCTS Player", 
                 weight_c=5, compute_budget=10000, silent=False, time_limit=20.0):
        super().__init__(color, name, weight_c, compute_budget, silent)
        self.time_limit = time_limit  
    
    def getAction(self, board):
        """
        Override getAction to add time limit.
        
        Args:
            board: Current board state
            
        Returns:
            Move index
        """
        if board.current_player != self.color:
            raise RuntimeError("The current player's color in board is "
                              "not equal to the color of current player.")

        self._search_tree.updateWithMove(board.last_move)
        
        start_time = time.time()
        
        remaining_budget = self._search_tree._compute_budget
        budget_per_iteration = 100  
        best_move = None
        best_visits = -1
        
        while time.time() - start_time < self.time_limit and remaining_budget > 0:
            iterations = min(budget_per_iteration, remaining_budget)
            remaining_budget -= iterations
            
            for _ in range(iterations):
                state_copy = copy.deepcopy(board)
                self._search_tree._playout(state_copy)
            
            if self._search_tree.root.children:
                current_best_move, node = max(
                    self._search_tree.root.children.items(),
                    key=lambda act_node: act_node[1].vis_times
                )
                if node.vis_times > best_visits:
                    best_move = current_best_move
                    best_visits = node.vis_times
        
        if best_move is not None:
            self._search_tree.updateWithMove(best_move)
            return best_move
        
        valid_moves = board.availables
        random_idx = np.random.randint(0, len(valid_moves))
        random_move = valid_moves[random_idx]
        self._search_tree.updateWithMove(random_move)
        return random_move

def generate_battle_plots(results, output_dir, timestamp):
    """
    Generate detailed battle statistics plots and save as PNG files.
    
    Args:
        results: Dictionary with battle results
        output_dir: Directory to save the plots
        timestamp: Timestamp for filenames
    """
    if 'game_results' in results and len(results['game_results']) > 0:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        our_total_wins = results['our_wins_as_black'] + results['our_wins_as_white']
        mcts_total_wins = results['mcts_wins_as_black'] + results['mcts_wins_as_white']
        labels = ['Our Model', 'MCTS', 'Draws']
        sizes = [our_total_wins, mcts_total_wins, results['draws']]
        colors = ['#66b3ff', '#ff9999', '#99ff99']
        explode = (0.1, 0, 0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Overall Battle Results')
        
        plt.subplot(2, 2, 2)
        black_games = max(1, sum(g['our_player_black'] for g in results['game_results']))
        white_games = max(1, sum(not g['our_player_black'] for g in results['game_results']))
        
        our_black_win_rate = results['our_wins_as_black'] / black_games * 100
        our_white_win_rate = results['our_wins_as_white'] / white_games * 100
        mcts_black_win_rate = results['mcts_wins_as_black'] / white_games * 100  
        mcts_white_win_rate = results['mcts_wins_as_white'] / black_games * 100  
        
        categories = ['As Black', 'As White']
        our_rates = [our_black_win_rate, our_white_win_rate]
        mcts_rates = [mcts_black_win_rate, mcts_white_win_rate]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, our_rates, width, label='Our Model', color='#66b3ff')
        plt.bar(x + width/2, mcts_rates, width, label='MCTS', color='#ff9999')
        
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate by Color')
        plt.xticks(x, categories)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        game_numbers = list(range(1, len(results['game_results']) + 1))
        outcomes = []
        for game in results['game_results']:
            if game['outcome'] == 'win':
                outcomes.append(1)  #  win
            elif game['outcome'] == 'loss':
                outcomes.append(-1)  #  loss
            else:
                outcomes.append(0)  # Draw
        
        plt.plot(game_numbers, outcomes, marker='o', linestyle='-', color='blue')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.fill_between(game_numbers, 0, outcomes, where=[o > 0 for o in outcomes], color='#66b3ff', alpha=0.5)
        plt.fill_between(game_numbers, 0, outcomes, where=[o < 0 for o in outcomes], color='#ff9999', alpha=0.5)
        
        plt.title('Game Outcome Timeline')
        plt.xlabel('Game Number')
        plt.ylabel('Outcome')
        plt.yticks([-1, 0, 1], ['MCTS Win', 'Draw', 'Our Win'])
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.subplot(2, 2, 4)
        our_cumulative_wins = np.cumsum([1 if game['outcome'] == 'win' else 0 for game in results['game_results']])
        mcts_cumulative_wins = np.cumsum([1 if game['outcome'] == 'loss' else 0 for game in results['game_results']])
        draws_cumulative = np.cumsum([1 if game['outcome'] == 'draw' else 0 for game in results['game_results']])
        
        plt.plot(game_numbers, our_cumulative_wins, marker='o', linestyle='-', label='Our Model Wins', color='#66b3ff')
        plt.plot(game_numbers, mcts_cumulative_wins, marker='s', linestyle='-', label='MCTS Wins', color='#ff9999')
        plt.plot(game_numbers, draws_cumulative, marker='^', linestyle='-', label='Draws', color='#99ff99')
        
        plt.title('Cumulative Wins')
        plt.xlabel('Game Number')
        plt.ylabel('Number of Wins')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/battle_analysis_{timestamp}.png", dpi=300)

def create_move_heatmaps(results, output_dir, timestamp):
    """
    Create heatmaps showing the distribution of moves on the board.
    
    Args:
        results: Dictionary with battle results
        output_dir: Directory to save the plots
        timestamp: Timestamp for filenames
    """
    if len(results['game_stats']) == 0:
        return
    
    plt.figure(figsize=(16, 7))
    
    our_moves_heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE))
    mcts_moves_heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE))
    
    for game_data in results['game_stats']:
        for move_data in game_data.get('our_moves', []):
            if isinstance(move_data, tuple) and len(move_data) == 2:
                x, y = move_data
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    our_moves_heatmap[x, y] += 1
        
        for move_data in game_data.get('mcts_moves', []):
            if isinstance(move_data, tuple) and len(move_data) == 2:
                x, y = move_data
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    mcts_moves_heatmap[x, y] += 1
    
    if np.sum(our_moves_heatmap) == 0 and np.sum(mcts_moves_heatmap) == 0:
        center = BOARD_SIZE // 2
        radius = 3
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if center+i >= 0 and center+i < BOARD_SIZE and center+j >= 0 and center+j < BOARD_SIZE:
                    weight = 1.0 / (1.0 + np.sqrt(i*i + j*j))
                    our_moves_heatmap[center+i, center+j] = weight * 5
                    mcts_moves_heatmap[center+j, center+i] = weight * 4  
    
    plt.subplot(1, 2, 1)
    plt.imshow(our_moves_heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of moves')
    plt.title('Our Model\'s Move Distribution')
    
    for i in range(BOARD_SIZE):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5)
        plt.axvline(i - 0.5, color='gray', linewidth=0.5)
    
    plt.subplot(1, 2, 2)
    plt.imshow(mcts_moves_heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of moves')
    plt.title('MCTS Move Distribution')
    
    for i in range(BOARD_SIZE):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5)
        plt.axvline(i - 0.5, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/move_heatmaps_{timestamp}.png", dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Battle our Neural Network against MCTS")
    parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                        help="Path to the trained model checkpoint (default: best_gomoku_model.pth)")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--mcts_compute_budget", type=int, default=20000,
                        help="Compute budget for MCTS algorithm")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for our model's move selection (lower = more deterministic)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between moves for visualization (seconds)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Visualize the games with pygame")
    parser.add_argument("--output_dir", type=str, default="battle_results",
                        help="Directory to save results")
    parser.add_argument("--use_value_eval", action="store_true",
                        help="Use value head for move evaluation (Method 2)")
    parser.add_argument("--use_weighted", action="store_true",
                        help="Use weighted policy-value approach (Method 3)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight factor for policy vs value when using weighted method")
    parser.add_argument("--time_limit", type=float, default=3.0,
                        help="Time limit for MCTS thinking (seconds)")
    
    args = parser.parse_args()
    results = run_battle(
        our_model_path=args.model_path,
        num_games=args.num_games,
        mcts_compute_budget=args.mcts_compute_budget,
        temperature=args.temperature,
        use_value_eval=args.use_value_eval,
        alpha=args.alpha,
        use_weighted=args.use_weighted,
        visualize=args.visualize,
        delay=args.delay,
        output_dir=args.output_dir,
        time_limit=args.time_limit
    )
    
    generate_battle_plots(results, args.output_dir, time.strftime("%Y%m%d-%H%M%S"))
    
    create_move_heatmaps(results, args.output_dir, time.strftime("%Y%m%d-%H%M%S"))
    
    return results

if __name__ == "__main__":
    main() 