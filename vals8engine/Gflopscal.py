import torch
import numpy as np
import time
import psutil
import os
import sys
import json
import matplotlib.pyplot as plt
from tabulate import tabulate
from thop import profile

# Adjust Python path to import from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # This should be C:\Users\User\Desktop\sgold
sys.path.insert(0, project_root) # Add project root to the beginning of sys.path

# Add opponent engine to path relative to project root
opponent_engine_dir = os.path.join(project_root, "Opponent_gomoku_engine")
sys.path.append(opponent_engine_dir)

# Import models from both engines
from model import GomokuNet
from inference import load_model, raw_board_to_tensor
try:
    from Opponent_gomoku_engine.ai import ConvNet, GomokuAI
    OPPONENT_ENGINE_AVAILABLE = True
except ImportError:
    print("Opponent engine not available, skipping opponent comparison")
    OPPONENT_ENGINE_AVAILABLE = False

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_flops(model, input_shape, device):
    """Measure FLOPs for a forward pass using thop"""
    try:
        input_tensor = torch.randn(*input_shape).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"Error measuring FLOPs: {e}")
        # Fallback to a simpler estimation based on parameter count
        # This part requires count_parameters, which was intended to be kept for fallback
        params = count_parameters(model) 
        # Rough estimate: each parameter is used in ~2 operations
        return params * 2 * input_shape[0], params


def generate_random_board(board_size=15, stone_count=20):
    """Generate a random board state with specified stone count"""
    board = np.zeros((board_size, board_size, 3), dtype=np.float32)
    board[:, :, 2] = 1.0  # Initialize all positions as empty
    
    stones_placed = 0
    while stones_placed < stone_count:
        row, col = np.random.randint(0, board_size), np.random.randint(0, board_size)
        if board[row, col, 2] == 1.0:  # If empty
            color = stones_placed % 2
            board[row, col, color] = 1.0
            board[row, col, 2] = 0.0
            stones_placed += 1
            
    return board

def benchmark_engine(model, device='cuda', num_iterations=100, board_size=15, is_opponent=False):
    """
    Benchmark engine performance
    
    Args:
        model: The model to benchmark
        device: 'cuda' or 'cpu'
        num_iterations: Number of iterations for benchmarking
        board_size: Size of the Gomoku board
        is_opponent: Whether this is the opponent engine
        
    Returns:
        A dictionary with benchmark results
    """
    model = model.to(device)
    
    print(f"\nMODEL STATS ({'Opponent' if is_opponent else 'Your'} Engine):")
    print(f"Device: {next(model.parameters()).device}")
    
    board_states = []
    for i in range(5):
        board = generate_random_board(board_size=board_size, stone_count=10*i)
        board_states.append(board)
    
    input_shape = (1, 3, board_size, board_size)
    # Ensure measure_flops uses the model on the correct device for FLOPs calculation
    flops, _ = measure_flops(model.to(device), input_shape, device) 
    gflops = flops / 1e9
    
    print(f"\nBENCHMARK RESULTS:")
    print(f"FLOPs per forward pass: {flops:,}")
    print(f"GFLOPs per forward pass: {gflops:.4f}")
    
    total_time = 0
    for idx, board in enumerate(board_states):
        if not is_opponent:
            board_tensor = raw_board_to_tensor(board).to(device)
        else:
            board_2d = np.zeros((board_size, board_size), dtype=np.int32)
            for r_idx in range(board_size):
                for c_idx in range(board_size):
                    if board[r_idx, c_idx, 0] == 1:
                        board_2d[r_idx, c_idx] = 1
                    elif board[r_idx, c_idx, 1] == 1:
                        board_2d[r_idx, c_idx] = 2
            
            ai = GomokuAI(_board_size=board_size)
            one_hot_board = ai.convert_to_one_hot(board_2d, player_id=1)
            board_tensor = torch.tensor(one_hot_board, dtype=torch.float32).unsqueeze(0).to(device)
        
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                if not is_opponent:
                    policy_logits, value_pred = model(board_tensor)
                else:
                    prediction = model(board_tensor)
        end_time = time.time()
        
        avg_time_per_board_iter = (end_time - start_time) / num_iterations * 1000
        gflops_per_second_this_board = gflops / (avg_time_per_board_iter / 1000)
        
        total_time += avg_time_per_board_iter
        stone_count = np.sum(board[:,:,0]) + np.sum(board[:,:,1]) if not is_opponent else np.sum(board_2d > 0)
        # Removed avg_time from this print statement
        print(f"Board {idx+1} with {stone_count} stones: {gflops_per_second_this_board:.2f} GFLOPS/s")
    
    avg_inference_time_ms = total_time / len(board_states) # Still needed for overall GFLOPS/s
    avg_gflops_per_second = gflops / (avg_inference_time_ms / 1000) # Calculated using the internal avg_inference_time_ms
    
    # print(f"Average inference time: {avg_inference_time_ms:.2f} ms") # This line is removed from output
    print(f"Average performance: {avg_gflops_per_second:.2f} GFLOPS/s")
    
    return {
        "model_type": "Opponent" if is_opponent else "Your",
        "flops": flops,
        "gflops": gflops,
        # "avg_inference_time_ms": avg_inference_time_ms, # Removed from returned dict
        "avg_gflops_per_second": avg_gflops_per_second
    }

def compare_engines(your_results, opponent_results=None):
    """Compare the results of your engine with opponent's"""
    headers = ["Metric", "Your Engine", "Opponent Engine", "Ratio (You/Opponent)"]
    
    if opponent_results:
        rows = [
            ["GFLOPs per inference", f"{your_results['gflops']:.4f}", f"{opponent_results['gflops']:.4f}", 
             f"{your_results['gflops'] / opponent_results['gflops']:.2f}x"],
            # ["Avg. Inference Time", ...], # Row removed
            ["GFLOPS/s", f"{your_results['avg_gflops_per_second']:.2f}", f"{opponent_results['avg_gflops_per_second']:.2f}", 
             f"{your_results['avg_gflops_per_second'] / opponent_results['avg_gflops_per_second']:.2f}x"]
        ]
    else:
        headers = ["Metric", "Your Engine"]
        rows = [
            ["GFLOPs per inference", f"{your_results['gflops']:.4f}"],
            # ["Avg. Inference Time", ...], # Row removed
            ["GFLOPS/s", f"{your_results['avg_gflops_per_second']:.2f}"]
        ]
    
    print("\nPERFORMANCE COMPARISON:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    results = {
        "your_engine": your_results,
        "opponent_engine": opponent_results if opponent_results else "Not available"
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"engine_performance_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results

def plot_performance_comparison(results):
    """Create performance comparison plots"""
    if "opponent_engine" not in results or results["opponent_engine"] == "Not available":
        labels = ['Your Engine']
        gflops_data = [results['your_engine']['gflops']]
        gflops_per_sec_data = [results['your_engine']['avg_gflops_per_second']]
    else:
        labels = ['Your Engine', 'Opponent Engine']
        gflops_data = [results['your_engine']['gflops'], results['opponent_engine']['gflops']]
        gflops_per_sec_data = [results['your_engine']['avg_gflops_per_second'], 
                               results['opponent_engine']['avg_gflops_per_second']]
    
    plt.figure(figsize=(8, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(labels, gflops_data, color=['blue', 'red'] if len(labels) > 1 else ['blue'])
    plt.title('GFLOPs per Inference')
    plt.ylabel('GFLOPs')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(labels, gflops_per_sec_data, color=['blue', 'red'] if len(labels) > 1 else ['blue'])
    plt.title('Computational Efficiency')
    plt.ylabel('GFLOPS/s')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_file = f"engine_performance_comparison_{timestamp}.png"
    plt.savefig(plot_file)
    print(f"Performance comparison plot saved to {plot_file}")
    
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark and compare Gomoku engines")
    parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                        help="Path to your model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for timing tests")
    parser.add_argument("--board_size", type=int, default=15,
                        help="Size of the Gomoku board")
    parser.add_argument("--skip_opponent", action="store_true",
                        help="Skip opponent comparison")
    args = parser.parse_args()
    
    print(f"ANALYZING ENGINE PERFORMANCE")
    print(f"===========================")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Iterations: {args.iterations}")
    print(f"Board size: {args.board_size}")
    
    print("\nLoading your model...")
    your_model = load_model(args.model_path, args.device)
    your_results = benchmark_engine(your_model, args.device, args.iterations, args.board_size, is_opponent=False)
    
    opponent_results = None
    if OPPONENT_ENGINE_AVAILABLE and not args.skip_opponent:
        print("\nLoading opponent model...")
        # Opponent model might need specific instantiation if its structure is known
        # Corrected instantiation based on Opponent_gomoku_engine/ai.py ConvNet definition
        opponent_model = ConvNet(input_dim=args.board_size, hidden_dim=32, output_dim=1) 
        opponent_results = benchmark_engine(opponent_model, args.device, args.iterations, args.board_size, is_opponent=True)
    
    results = compare_engines(your_results, opponent_results)
    
    try:
        plot_performance_comparison(results)
    except Exception as e:
        print(f"Error creating performance plot: {e}")

if __name__ == "__main__":
    main() 