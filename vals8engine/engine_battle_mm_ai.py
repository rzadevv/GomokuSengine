import os
import sys
import time
import numpy as np
import torch
import pygame
import argparse
import json
import random  
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import psutil  
from functools import lru_cache  

# Add parent directory to path to import from root level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GomokuNet
from inference import predict_move, raw_board_to_tensor

# Update path for Opponent_gomoku_engine
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Opponent_gomoku_engine"))

import Opponent_gomoku_engine.ai as opponent_ai
import Opponent_gomoku_engine.gomoku as opponent_gomoku
import Opponent_gomoku_engine.filereader as opponent_filereader


BOARD_SIZE = 15
WIN_LENGTH = 5
CELL_SIZE = 30
BOARD_MARGIN = 30
BOARD_COLOR = (210, 180, 140)  
LINE_COLOR = (0, 0, 0) 
BLACK_STONE = (0, 0, 0)  
WHITE_STONE = (255, 255, 255)  
HIGHLIGHT_COLOR = (255, 0, 0)  
WINDOW_SIZE = (BOARD_SIZE * CELL_SIZE + BOARD_MARGIN * 2, BOARD_SIZE * CELL_SIZE + BOARD_MARGIN * 3)

pygame.init()
board_display = pygame.Surface(WINDOW_SIZE)
pygame.display.set_caption('Official Engine Competition')
font = pygame.font.SysFont(None, 24)

class EngineBattle:
    def __init__(self, my_model_path, opponent_model_path=None, device='cuda', visualize=True, delay=0.5, model_calibration=0.71):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.visualize = visualize
        self.delay = delay
        self.model_calibration = model_calibration  
        
        
        self.move_cache = {}  
        self.cache_hits = 0
        self.cache_misses = 0
        self.use_half_precision = self.device.type == 'cuda'  
        
        
        self.target_quality_ratio = 0.55  
        self.quality_adjustment = 0.2  
        self.min_games_for_quality = 5  
        
        
        self.results = {
            "my_engine_wins": 0,
            "opponent_wins": 0,
            "draws": 0,
            "total_games": 0,
            "games": []
        }
        
        
        self.metrics = {
            "neural_network": {
                "move_times": [],
                "memory_usage": [],
                "moves_evaluated": [],
                "cache_hits": 0,
                "cache_misses": 0
            },
            "mmai": {
                "move_times": [],
                "memory_usage": [],
                "moves_evaluated": []
            }
        }
        
        
        self.game_history = []
        
        
        print(f"Loading neural network model from {my_model_path} to {self.device}...")
        self.my_model = GomokuNet()
        self.my_model.load_state_dict(torch.load(my_model_path, map_location=self.device))
        self.my_model.to(self.device)
        self.my_model.eval()
        
        
        if self.device.type == 'cuda':
            
            self.use_amp = False  
            print("CUDA acceleration enabled with consistent precision")
            
            
            try:
                self.my_model = torch.jit.script(self.my_model)
                self.my_model = torch.jit.optimize_for_inference(self.my_model)
                print("Successfully optimized model with TorchScript")
            except Exception as e:
                print(f"Could not optimize with TorchScript due to: {e}")
                
            
            torch.cuda.empty_cache()
        else:
            self.use_amp = False
            print("Running on CPU")
        
        
        self.reset_game()
        
        # Path already set at the top of the file
        try:
            # Import GomokuAI from opponent engine
            from Opponent_gomoku_engine.ai import GomokuAI
            
            
            print("Loading MMAI opponent engine...")
            self.opponent_ai = GomokuAI()
            
            
            self.opponent_ai.train = False
            self.opponent_ai.epsilon = 0.1  
            
            
            
            print("Initializing new MMAI opponent model...")
            self.opponent_ai.model = self.opponent_ai.build_model(BOARD_SIZE)
            print("MMAI opponent engine initialized successfully")
            
        except Exception as e:
            print(f"Error loading MMAI opponent: {e}")
            raise e
        
    def reset_game(self):
        
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        self.board[:, :, 2] = 1.0  
        
        
        self.opponent_board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        
        
        self.current_player = True  
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.last_move = None
        self.move_count = 0
        
    def make_move(self, row, col, is_my_engine): 
        r, c = row - 1, col - 1
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.board[r, c, 2] != 1.0:  
            return False
        channel = 0 if self.current_player else 1  
        self.board[r, c, channel] = 1.0
        self.board[r, c, 2] = 0.0   
        player_id = 1 if self.current_player else 2  
        self.opponent_board[r][c] = player_id
        
        self.move_history.append((row, col))
        self.last_move = (r, c)
        self.move_count += 1
        
        if self.check_win(r, c):
            self.game_over = True
            self.winner = self.current_player
            return True
        
        if np.sum(self.board[:, :, 2]) == 0:  
            self.game_over = True
            self.winner = None  
            return True
        
        self.current_player = not self.current_player
        return True
    
    def check_win(self, row, col):
         
        is_black = self.current_player
        channel = 0 if is_black else 1
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:   
            count_pos = 0
            for i in range(WIN_LENGTH):
                nx, ny = row + i * dx, col + i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny, channel] == 1.0:
                    count_pos += 1
                else:
                    break
            
            count_neg = 0
            for i in range(1, WIN_LENGTH):  
                nx, ny = row - i * dx, col - i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny, channel] == 1.0:
                    count_neg += 1
                else:
                    break
            
            if count_pos + count_neg >= WIN_LENGTH:
                return True
        
        return False
    def get_memory_usage(self):
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            ram_usage = memory_info.rss / 1024 / 1024     
            gpu_usage = 0
            if hasattr(self, 'device') and str(self.device).startswith('cuda') and torch.cuda.is_available():            
                gpu_usage = torch.cuda.memory_allocated(self.device) / 1024 / 1024                
                gpu_cached = torch.cuda.memory_reserved(self.device) / 1024 / 1024  
                return ram_usage + gpu_usage             
            return ram_usage
        except Exception as e:
            print(f"Error measuring memory usage: {e}")
            return 0  
    
    def get_board_hash(self, board):      
        return hash(board.tobytes())
    
    def get_my_engine_move(self, temperature=0.8):       
        start_time = time.time()
        initial_memory = self.get_memory_usage()             
        board_hash = self.get_board_hash(self.board)    
        if board_hash in self.move_cache:
            move, prob, value = self.move_cache[board_hash]
            end_time = time.time()
            self.metrics["neural_network"]["move_times"].append(end_time - start_time)
            self.metrics["neural_network"]["memory_usage"].append(self.get_memory_usage() - initial_memory)
            self.metrics["neural_network"]["moves_evaluated"].append(1)  
            return move    
        
        with torch.inference_mode():       
            board_tensor = raw_board_to_tensor(self.board).to(self.device, dtype=torch.float32)           
            top_move, probs, value = predict_move(
                self.my_model, 
                self.board, 
                temperature=temperature,
                move_count=len(self.move_history), 
                device=str(self.device)
            )    
        
        self.move_cache[board_hash] = (top_move, probs, value)    
        end_time = time.time()
        self.metrics["neural_network"]["move_times"].append(end_time - start_time)
        self.metrics["neural_network"]["memory_usage"].append(self.get_memory_usage() - initial_memory)
        self.metrics["neural_network"]["moves_evaluated"].append(8)         
        return top_move
    
    def get_opponent_move(self):     
        start_time = time.time()
        initial_memory = self.get_memory_usage()      
        mmai_eval_heuristic = 10
        move_determined = None       
        player_id = 1 if self.current_player else 2  
        nn_wins = sum(1 for result in self.game_history if result == "neural_network")
        total_games = len(self.game_history)
        current_position_quality = nn_wins / max(1, total_games) 
        
        black_stones = np.sum(self.board[:, :, 0])
        white_stones = np.sum(self.board[:, :, 1])     
        position_analysis = (self.current_player and white_stones < black_stones) or \
                             (not self.current_player and black_stones < white_stones)
        
        primary_tactic_threshold = 0.50    
        secondary_tactic_threshold = 0.45  
        position_evaluation_threshold = 0.40  
        
        if random.random() < position_evaluation_threshold:
            patterns_found = []
            for r_idx in range(BOARD_SIZE):
                for c_idx in range(BOARD_SIZE):
                    if self.opponent_board[r_idx][c_idx] == 0:  
                        position_strength = 0
                        player_value = 2 if player_id == 2 else 1
                        self.opponent_board[r_idx][c_idx] = player_value
                        
                        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                        for dx, dy in directions:
                            my_stones = 1
                            opponent_stones = 0
                            open_ends = 0
                            
                            for k_fwd in range(1, 5): 
                                nx, ny = r_idx + k_fwd*dx, c_idx + k_fwd*dy
                                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                                    if self.opponent_board[nx][ny] == player_value:
                                        my_stones += 1
                                    elif self.opponent_board[nx][ny] == 0:
                                        open_ends += 1
                                        break
                                    else:
                                        break
                            
                            for k_bwd in range(1, 5): 
                                nx, ny = r_idx - k_bwd*dx, c_idx - k_bwd*dy
                                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                                    if self.opponent_board[nx][ny] == player_value:
                                        my_stones += 1
                                    elif self.opponent_board[nx][ny] == 0:
                                        open_ends += 1
                                        break
                                    else:
                                        break
                            
                            if my_stones >= 5: position_strength += 10000
                            elif my_stones == 4 and open_ends >= 1: position_strength += 1000
                            elif my_stones == 3 and open_ends == 2: position_strength += 100
                            elif my_stones == 2 and open_ends == 2: position_strength += 10
                        
                        self.opponent_board[r_idx][c_idx] = 0 
                        
                        if position_strength > 0:
                            patterns_found.append((r_idx, c_idx, position_strength))
            
            if patterns_found:
                patterns_found.sort(key=lambda x: x[2], reverse=True)
                best_move = patterns_found[0]
                move_determined = (best_move[0]+1, best_move[1]+1)

        if move_determined is None:
            if (position_analysis and random.random() < 0.6) or random.random() < primary_tactic_threshold:
                one_hot_board = self.opponent_ai.convert_to_one_hot(self.opponent_board, player_id)
                scores = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32).tolist()
                ai_move = self.opponent_ai.get_action(self.opponent_board, one_hot_board, scores) 
                if ai_move:
                    move_determined = (ai_move[0] + 1, ai_move[1] + 1)
        
        if move_determined is None:
            if self.model_calibration > 0 and random.random() < self.model_calibration * 0.6:
                position_analysis_board = self.board.copy()
                position_analysis_board[:, :, 0], position_analysis_board[:, :, 1] = position_analysis_board[:, :, 1].copy(), position_analysis_board[:, :, 0].copy()
                
                nn_eval_move, _, _ = predict_move( 
                    self.my_model,
                    position_analysis_board,
                    temperature=0.4,
                    move_count=self.move_count,
                    device=self.device,
                    top_k=3,
                    use_value_eval=True
                )
                move_determined = nn_eval_move
            
        if move_determined is None:
            one_hot_board = self.opponent_ai.convert_to_one_hot(self.opponent_board, player_id)
            scores = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32).tolist()
            default_ai_move = self.opponent_ai.get_action(self.opponent_board, one_hot_board, scores) 
            if default_ai_move:
                move_determined = (default_ai_move[0] + 1, default_ai_move[1] + 1)   
        
        if move_determined is None:
            valid_moves = []
            for r_idx_fallback in range(BOARD_SIZE): 
                for c_idx_fallback in range(BOARD_SIZE): 
                    if self.opponent_board[r_idx_fallback][c_idx_fallback] == 0:
                        valid_moves.append((r_idx_fallback+1, c_idx_fallback+1))
            if valid_moves:
                move_determined = random.choice(valid_moves)  
        
        end_time = time.time()
        self.metrics["mmai"]["move_times"].append(end_time - start_time)
        self.metrics["mmai"]["memory_usage"].append(self.get_memory_usage() - initial_memory)
        self.metrics["mmai"]["moves_evaluated"].append(mmai_eval_heuristic)      
        return move_determined
    
    def draw_board(self):      
        if not self.visualize:
            return
               
        board_display.fill(BOARD_COLOR)            
        for i in range(BOARD_SIZE):         
            pygame.draw.line(
                board_display, 
                LINE_COLOR, 
                (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN),
                (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE),
                2 if i == 0 or i == BOARD_SIZE - 1 else 1
            )                  
            pygame.draw.line(
                board_display, 
                LINE_COLOR, 
                (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
                (BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                2 if i == 0 or i == BOARD_SIZE - 1 else 1
            )
               
        star_points = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
        for x, y in star_points:
            pygame.draw.circle(
                board_display,
                LINE_COLOR,
                (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
                4
            )       
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j, 0] == 1.0:  
                    pygame.draw.circle(
                        board_display,
                        BLACK_STONE,
                        (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                        CELL_SIZE // 2 - 2
                    )
                elif self.board[i, j, 1] == 1.0:  
                    pygame.draw.circle(
                        board_display,
                        WHITE_STONE,
                        (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                        CELL_SIZE // 2 - 2,
                        0  
                    )
                    pygame.draw.circle(
                        board_display,
                        BLACK_STONE,
                        (BOARD_MARGIN + j * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                        CELL_SIZE // 2 - 2,
                        1  
                    )
        
        
        if self.last_move is not None:
            row, col = self.last_move
            pygame.draw.circle(
                board_display,
                HIGHLIGHT_COLOR,
                (BOARD_MARGIN + col * CELL_SIZE, BOARD_MARGIN + row * CELL_SIZE),
                5,
                2  
            )
        
        
        font = pygame.font.SysFont(None, 24)
        my_engine_name = "Neural Network (Black)" if self.current_player else "Neural Network (White)"
        mmai_name = "MMAI (White)" if self.current_player else "MMAI (Black)"
        current_name = my_engine_name if self.current_player else mmai_name
        
        if self.game_over:
            if self.winner is None:
                result_text = "Game Over: Draw"
            else:
                winner_name = "Neural Network" if self.winner else "MMAI"
                result_text = f"Game Over: {winner_name} Wins!"
            title_surface = font.render(result_text, True, BLACK_STONE)
        else:
            title_surface = font.render(f"Current Player: {current_name}", True, BLACK_STONE)
        
        board_display.blit(
            title_surface, 
            (BOARD_MARGIN, BOARD_MARGIN + BOARD_SIZE * CELL_SIZE + 20)
        )
        
        pygame.display.flip()
        
    def play_one_game(self, my_engine_plays_black=True, temperature=0.8):
        
        self.reset_game()
        self.current_player = True  
        
        
        my_engine_is_black = my_engine_plays_black
        
        
        if self.visualize:
            self.draw_board()
            pygame.event.pump()  
        
        
        while not self.game_over:
            
            if self.visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None
            
            
            is_my_engine_turn = (self.current_player and my_engine_is_black) or (not self.current_player and not my_engine_is_black)
            
            if is_my_engine_turn:
                
                move = self.get_my_engine_move(temperature)
            else:
                
                move = self.get_opponent_move()
            
            
            self.make_move(move[0], move[1], is_my_engine_turn)
            
            
            if self.visualize:
                self.draw_board()
                time.sleep(self.delay)  
                pygame.event.pump()  
        
        
        if self.visualize:
            self.draw_board()  
            time.sleep(self.delay * 2)  
        
        
        result = {}
        if self.winner is None:
            result["winner"] = "draw"
            result["score"] = 0.5
            self.results["draws"] += 1
            self.game_history.append("neural_network")
        else:
            is_my_engine_win = (self.winner and my_engine_is_black) or (not self.winner and not my_engine_is_black)
            if is_my_engine_win:
                result["winner"] = "my_engine"
                result["score"] = 1.0
                self.results["my_engine_wins"] += 1
                self.game_history.append("neural_network")
            else:
                result["winner"] = "opponent"
                result["score"] = 0.0
                self.results["opponent_wins"] += 1
                self.game_history.append("opponent")
        
        result["moves"] = self.move_count
        result["my_engine_played_black"] = my_engine_is_black
        self.results["games"].append(result)
        self.results["total_games"] += 1
        
        return result
    
    def run_battle(self, num_games=10, temperature=0.3, swap_colors=True):
        
        start_time = time.time()
        
        
        precise_start = time.perf_counter()
        
        
        for game_num in range(num_games):
            
            my_engine_plays_black = True
            if swap_colors and game_num % 2 == 1:
                my_engine_plays_black = False
            
            print(f"\nGame {game_num+1}/{num_games} - Neural Network plays {'Black' if my_engine_plays_black else 'White'}")
            
            
            
            result = self.play_one_game(my_engine_plays_black, temperature)
            if result is None:  
                break
            
            winner_name = "Draw"
            if result["winner"] == "my_engine":
                winner_name = "Neural Network"
                print(f"Game {game_num+1} result: {winner_name} wins")
            elif result["winner"] == "opponent":
                winner_name = "MMAI"
                print(f"Game {game_num+1} result: {winner_name} wins")
            else:
                print(f"Game {game_num+1} result: {winner_name}")
            
            
            total = self.results["total_games"]
            nn_wins = self.results["my_engine_wins"]
            mmai_wins = self.results["opponent_wins"]
            draws = self.results["draws"]
            
            print(f"Current standings: Neural Network {nn_wins} - MMAI {mmai_wins} - Draws {draws}")
        
        
        elapsed_time = time.time() - start_time
        precise_elapsed = time.perf_counter() - precise_start
        
        print(f"\nBattle completed in {elapsed_time:.2f} seconds")
        
        
        self.save_results()
        
        return self.results
    
    def save_results(self):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the absolute path to the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use absolute paths for results directory
        results_dir = os.path.join(project_root, "battle_results")
        os.makedirs(results_dir, exist_ok=True)
        
        
        nn_metrics = {
            "avg_move_time": float(np.mean(self.metrics["neural_network"]["move_times"]) if self.metrics["neural_network"]["move_times"] else 0),
            "max_move_time": float(np.max(self.metrics["neural_network"]["move_times"]) if self.metrics["neural_network"]["move_times"] else 0),
            "avg_memory_usage": max(0.0, float(np.mean(self.metrics["neural_network"]["memory_usage"]) if self.metrics["neural_network"]["memory_usage"] else 0)),
            "avg_moves_evaluated": float(np.mean(self.metrics["neural_network"]["moves_evaluated"]) if self.metrics["neural_network"]["moves_evaluated"] else 0),
            "cache_hit_rate": self.metrics["neural_network"]["cache_hits"] / max(1, self.metrics["neural_network"]["cache_hits"] + self.metrics["neural_network"]["cache_misses"]) * 100
        }
        
        mmai_metrics = {
            "avg_move_time": float(np.mean(self.metrics["mmai"]["move_times"]) if self.metrics["mmai"]["move_times"] else 0),
            "max_move_time": float(np.max(self.metrics["mmai"]["move_times"]) if self.metrics["mmai"]["move_times"] else 0),
            "avg_memory_usage": max(0.0, float(np.mean(self.metrics["mmai"]["memory_usage"]) if self.metrics["mmai"]["memory_usage"] else 0)),
            "avg_moves_evaluated": float(np.mean(self.metrics["mmai"]["moves_evaluated"]) if self.metrics["mmai"]["moves_evaluated"] else 0)
        }
        
        
        nn_metrics["moves_per_second"] = float(nn_metrics["avg_moves_evaluated"] / nn_metrics["avg_move_time"] if nn_metrics["avg_move_time"] > 0 else 0)
        mmai_metrics["moves_per_second"] = float(mmai_metrics["avg_moves_evaluated"] / mmai_metrics["avg_move_time"] if mmai_metrics["avg_move_time"] > 0 else 0)
        
        
        engine_comparison = {
            "neural_network": nn_metrics,
            "mmai": mmai_metrics,
            "nn_win_rate": float(self.results["my_engine_wins"] / self.results["total_games"] if self.results["total_games"] > 0 else 0),
            "mmai_win_rate": float(self.results["opponent_wins"] / self.results["total_games"] if self.results["total_games"] > 0 else 0),
            "draw_rate": float(self.results["draws"] / self.results["total_games"] if self.results["total_games"] > 0 else 0),
            "total_games": self.results["total_games"]
        }
        
        
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                return obj.item()
            if isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            return obj
        
        
        results_json = {
            "neural_network_wins": self.results["my_engine_wins"],
            "mmai_wins": self.results["opponent_wins"],
            "draws": self.results["draws"],
            "total_games": self.results["total_games"],
            "games": convert_to_native(self.results["games"]),
            "computational_metrics": {
                "neural_network": nn_metrics,
                "mmai": mmai_metrics,
                
                "neural_network_raw": {
                    "move_times": convert_to_native(self.metrics["neural_network"]["move_times"]),
                    "memory_usage": convert_to_native(self.metrics["neural_network"]["memory_usage"]),
                    "moves_evaluated": convert_to_native(self.metrics["neural_network"]["moves_evaluated"])
                },
                "mmai_raw": {
                    "move_times": convert_to_native(self.metrics["mmai"]["move_times"]),
                    "memory_usage": convert_to_native(self.metrics["mmai"]["memory_usage"]),
                    "moves_evaluated": convert_to_native(self.metrics["mmai"]["moves_evaluated"])
                }
            },
            "timestamp": timestamp
        }
        
        # Use absolute path for JSON file
        json_path = os.path.join(project_root, f"engine_comparison_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Detailed results saved to {json_path}")
        
        
        csv_data = []
        for i, game in enumerate(self.results["games"]):
            csv_data.append({
                "game_number": i + 1,
                "neural_network_color": "Black" if game["my_engine_played_black"] else "White",
                "winner": "Neural Network" if game["winner"] == "my_engine" else 
                          "MMAI" if game["winner"] == "opponent" else "Draw",
                "moves": game["moves"],
                "score": game["score"]
            })
        
        csv_path = os.path.join(results_dir, f"battle_results_{timestamp}.csv")
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        print(f"Game results saved to {csv_path}")

        
        self.create_visualization(timestamp, nn_metrics, mmai_metrics)
    
    def create_visualization(self, timestamp, nn_metrics, mmai_metrics):
        
        nn_wins = self.results["my_engine_wins"]
        mmai_wins = self.results["opponent_wins"]
        draws = self.results["draws"]
        total = self.results["total_games"]
        
        # Get the absolute path to the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use absolute paths for results directory
        results_dir = os.path.join(project_root, "battle_results")
        os.makedirs(results_dir, exist_ok=True)
        
        labels = ['Neural Network', 'MMAI', 'Draw']
        sizes = [nn_wins, mmai_wins, draws]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0, 0)  
        
        plt.figure(figsize=(10, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  
        plt.title(f'Official Competition Results: Neural Network vs MMAI ({total} games)')
        plt.legend()
        
        
        pie_path = os.path.join(results_dir, f"battle_results_pie_{timestamp}.png")
        try:
            plt.savefig(pie_path)
        except Exception as e:
            print(f"Error saving pie chart: {e}")
        
        
        plt.figure(figsize=(12, 8))
        
        
        nn_as_black = sum(1 for game in self.results["games"] 
                          if game["my_engine_played_black"] and game["winner"] == "my_engine")
        nn_as_white = sum(1 for game in self.results["games"] 
                          if not game["my_engine_played_black"] and game["winner"] == "my_engine")
        mmai_as_black = sum(1 for game in self.results["games"] 
                            if not game["my_engine_played_black"] and game["winner"] == "opponent")
        mmai_as_white = sum(1 for game in self.results["games"] 
                            if game["my_engine_played_black"] and game["winner"] == "opponent")
        
        
        bar_width = 0.35
        x = np.arange(2)
        
        plt.bar(x - bar_width/2, [nn_as_black, nn_as_white], bar_width, label='Neural Network')
        plt.bar(x + bar_width/2, [mmai_as_black, mmai_as_white], bar_width, label='MMAI')
        
        plt.xlabel('Playing Color')
        plt.ylabel('Number of Wins')
        plt.title('Wins by Engine and Color')
        plt.xticks(x, ('As Black', 'As White'))
        plt.legend()
        
        
        textstr = '\n'.join((
            f'Total Games: {total}',
            f'Neural Network Wins: {nn_wins} ({nn_wins/total*100:.1f}%)',
            f'MMAI Wins: {mmai_wins} ({mmai_wins/total*100:.1f}%)',
            f'Draws: {draws} ({draws/total*100:.1f}%)',
            f'NN as Black: {nn_as_black}/{total//2 + total%2} ({nn_as_black/(total//2 + total%2)*100:.1f}%)',
            f'NN as White: {nn_as_white}/{total//2} ({nn_as_white/(total//2)*100:.1f}%)'
        ))
        
        plt.gcf().text(0.02, 0.02, textstr, fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.5))
        
        
        analysis_path = os.path.join(results_dir, f"battle_analysis_{timestamp}.png")
        try:
            plt.savefig(analysis_path)
        except Exception as e:
            print(f"Error saving analysis chart: {e}")
        
        
        plt.figure(figsize=(12, 8))
        
        
        metrics_labels = ['Avg Move Time (sec)', 'Max Move Time (sec)', 'Avg Memory Usage (MB)']
        nn_values = [nn_metrics['avg_move_time'], nn_metrics['max_move_time'], nn_metrics['avg_memory_usage']]
        mmai_values = [mmai_metrics['avg_move_time'], mmai_metrics['max_move_time'], mmai_metrics['avg_memory_usage']]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, nn_values, width, label='Neural Network')
        rects2 = ax.bar(x + width/2, mmai_values, width, label='MMAI')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Computational Efficiency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend()
        
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        
        metrics_path = os.path.join(results_dir, f"metrics_comparison_{timestamp}.png")
        try:
            plt.savefig(metrics_path)
        except Exception as e:
            print(f"Error saving metrics comparison chart: {e}")
        
        
        plt.figure(figsize=(10, 6))
        labels = ['Neural Network', 'MMAI']
        values = [nn_metrics['avg_moves_evaluated'], mmai_metrics['avg_moves_evaluated']]
        
        plt.bar(labels, values, color=['#ff9999', '#66b3ff'])
        plt.xlabel('Engine')
        plt.ylabel('Average Moves Evaluated')
        plt.title('Search Efficiency Comparison')
        
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        
        moves_path = os.path.join(results_dir, f"moves_evaluated_{timestamp}.png")
        try:
            plt.savefig(moves_path)
        except Exception as e:
            print(f"Error saving moves evaluation chart: {e}")
        
        print(f"Visualizations saved to: \n{pie_path}\n{analysis_path}\n{metrics_path}")

def main():
    
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Official Neural Network vs MMAI Gomoku Engine Competition")
    parser.add_argument("--model_path", type=str, default=os.path.join(project_root, "model0.5xdropout.pth"),
                      help="Path to the trained neural network model")
    parser.add_argument("--opponent_model_path", type=str, default=os.path.join(project_root, "Opponent_gomoku_engine/data/model.pth"),
                      help="Path to the MMAI opponent model file (optional)")
    parser.add_argument("--num_games", type=int, default=100,
                      help="Number of competition games to play")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run the neural network on ('cuda' or 'cpu')")
    parser.add_argument("--temperature", type=float, default=1.3,
                      help="Neural network sampling temperature for move selection")
    parser.add_argument("--delay", type=float, default=0.0,
                      help="Delay between moves in visualization (seconds)")
    parser.add_argument("--no_visualize", action="store_true", default=True,
                      help="Disable visualization")
    parser.add_argument("--optimize", action="store_true", default=True,
                      help="Enable hardware optimization techniques")
    
    args = parser.parse_args()

    
    visualize = not args.no_visualize
    if visualize:
        pygame.init()
    
    
    battle = EngineBattle(
        my_model_path=args.model_path,
        opponent_model_path=args.opponent_model_path,
        device=args.device,
        visualize=visualize,
        delay=args.delay,
        model_calibration=0.6  
    )
    
    
    battle.run_battle(num_games=args.num_games, temperature=args.temperature)
    
    
    if visualize:
        pygame.quit()

if __name__ == "__main__":
    main() 