import os
import time
import psutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import ai
from ai import GomokuAI
import gomoku
import mainmenu
import filereader

class TrainingCostMonitor:
    def __init__(self, sampling_interval=1.0):
        """
        Initialize the training cost monitor
        
        Args:
            sampling_interval: How often to sample metrics (in seconds)
        """
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.start_time = None
        self.end_time = None
        
        # Metrics storage
        self.cpu_percent = []
        self.memory_usage = []
        self.timestamp = []
        self.batch_sizes = []
        self.losses = []
        
        # Create results directory if it doesn't exist
        os.makedirs("training_metrics", exist_ok=True)
        
    def start_monitoring(self):
        """Start monitoring the training process"""
        self.start_time = time.time()
        self.cpu_percent = []
        self.memory_usage = []
        self.timestamp = []
        
    def sample_metrics(self):
        """Take a single sample of the metrics"""
        try:
            cpu = self.process.cpu_percent(interval=0.1)
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            
            current_time = time.time() - self.start_time
            
            self.cpu_percent.append(cpu)
            self.memory_usage.append(memory_mb)
            self.timestamp.append(current_time)
            
            return cpu, memory_mb, current_time
        except:
            return None, None, None
            
    def stop_monitoring(self):
        """Stop monitoring and calculate final statistics"""
        self.end_time = time.time()
        
    def log_batch_metrics(self, batch_size, loss):
        """Log metrics for a specific training batch"""
        self.batch_sizes.append(batch_size)
        self.losses.append(loss)
        
    def get_total_training_time(self):
        """Get total training time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
        
    def get_average_metrics(self):
        """Calculate average metrics"""
        avg_cpu = np.mean(self.cpu_percent) if self.cpu_percent else 0
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        peak_memory = max(self.memory_usage) if self.memory_usage else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        
        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "peak_memory_mb": peak_memory,
            "total_time_seconds": self.get_total_training_time(),
            "avg_loss": avg_loss
        }
        
    def save_metrics(self, filename_prefix):
        """Save metrics to files"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = self.get_average_metrics()
        
        # Save summary to text file
        summary_filename = f"training_metrics/{filename_prefix}_summary_{timestamp_str}.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"Training Cost Metrics\n")
            f.write(f"====================\n\n")
            f.write(f"Total training time: {metrics['total_time_seconds']:.2f} seconds\n")
            f.write(f"Average CPU usage: {metrics['avg_cpu_percent']:.2f}%\n")
            f.write(f"Average memory usage: {metrics['avg_memory_mb']:.2f} MB\n")
            f.write(f"Peak memory usage: {metrics['peak_memory_mb']:.2f} MB\n")
            f.write(f"Average loss: {metrics['avg_loss']:.6f}\n")
        
        # Save raw data to CSV
        csv_filename = f"training_metrics/{filename_prefix}_raw_{timestamp_str}.csv"
        with open(csv_filename, 'w') as f:
            f.write("time_seconds,cpu_percent,memory_mb\n")
            for t, cpu, mem in zip(self.timestamp, self.cpu_percent, self.memory_usage):
                f.write(f"{t:.2f},{cpu:.2f},{mem:.2f}\n")
        
        # Generate plots
        self._generate_plots(filename_prefix, timestamp_str)
        
        return summary_filename, csv_filename
        
    def _generate_plots(self, filename_prefix, timestamp_str):
        """Generate plots of the metrics"""
        # CPU Usage Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamp, self.cpu_percent)
        plt.title("CPU Usage During Training")
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.grid(True)
        plt.savefig(f"training_metrics/{filename_prefix}_cpu_{timestamp_str}.png")
        
        # Memory Usage Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamp, self.memory_usage)
        plt.title("Memory Usage During Training")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.grid(True)
        plt.savefig(f"training_metrics/{filename_prefix}_memory_{timestamp_str}.png")
        
        # Loss Plot (if we have loss data)
        if self.losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses)
            plt.title("Training Loss")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(f"training_metrics/{filename_prefix}_loss_{timestamp_str}.png")


def monkey_patch_ai_for_monitoring(monitor):
    """
    Monkey patch the AI class to monitor training metrics
    """
    # Store the original train_long_memory method
    original_train_long_memory = ai.GomokuAI.train_long_memory
    
    # Create a wrapper function to monitor metrics
    def monitored_train_long_memory(self):
        result = original_train_long_memory(self)
        
        # Log batch size and loss
        if hasattr(self, 'memory') and hasattr(self, 'loss'):
            monitor.log_batch_metrics(
                batch_size=len(self.memory) if len(self.memory) < ai.BATCH_SIZE else ai.BATCH_SIZE,
                loss=float(self.loss) if hasattr(self.loss, 'item') else self.loss
            )
            
        return result
    
    # Replace the original method with our monitored version
    ai.GomokuAI.train_long_memory = monitored_train_long_memory


def measure_training_cost(
    num_games=100,
    sampling_interval=1.0,
    player1_type="MM-AI",
    player2_type="TestAI"
):
    """
    Measure the cost of training the AI
    
    Args:
        num_games: Number of games to play for training
        sampling_interval: How often to sample metrics (in seconds)
        player1_type: Type of player 1 (should be MM-AI for training)
        player2_type: Type of player 2
    """
    # Create the monitor
    monitor = TrainingCostMonitor(sampling_interval)
    
    # Monkey patch the AI to monitor training metrics
    monkey_patch_ai_for_monitoring(monitor)
    
    # Setup game values
    values = filereader.create_gomoku_game("consts.json")
    game_instance = gomoku.GomokuGame(values)
    
    # Configure players
    gomoku.players[0].set_player(player1_type, 0)
    gomoku.players[1].set_player(player2_type, 1)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Start monitoring thread
    import threading
    stop_monitoring = [False]
    
    def monitoring_thread():
        while not stop_monitoring[0]:
            monitor.sample_metrics()
            time.sleep(sampling_interval)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()
    
    try:
        # Run the training
        print(f"Starting training for {num_games} games...")
        print(f"Player 1: {player1_type}, Player 2: {player2_type}")
        
        for i in range(num_games):
            if i % 10 == 0:
                print(f"Game {i+1}/{num_games}")
                
            game_instance.current_game = i+1
            game_instance.last_round = (i+1 == num_games)
            gomoku.run(game_instance, i, True, False, None)
            
    finally:
        # Stop monitoring
        stop_monitoring[0] = True
        thread.join(timeout=1.0)
        monitor.stop_monitoring()
        
    # Save metrics
    summary_file, csv_file = monitor.save_metrics("training")
    
    # Print summary
    metrics = monitor.get_average_metrics()
    print("\nTraining Cost Summary:")
    print(f"Total training time: {metrics['total_time_seconds']:.2f} seconds")
    print(f"Average CPU usage: {metrics['avg_cpu_percent']:.2f}%")
    print(f"Average memory usage: {metrics['avg_memory_mb']:.2f} MB")
    print(f"Peak memory usage: {metrics['peak_memory_mb']:.2f} MB")
    print(f"Average loss: {metrics['avg_loss']:.6f}")
    print(f"\nDetailed metrics saved to: {summary_file}")
    print(f"Raw data saved to: {csv_file}")
    
    return metrics, monitor


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Measure training cost of Gomoku AI")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play for training")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--player1", type=str, default="MM-AI", help="Type of player 1")
    parser.add_argument("--player2", type=str, default="TestAI", help="Type of player 2")
    
    args = parser.parse_args()
    
    # Measure training cost
    measure_training_cost(
        num_games=args.games,
        sampling_interval=args.interval,
        player1_type=args.player1,
        player2_type=args.player2
    ) 