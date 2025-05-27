import logging
import os
import time
import yaml
from datetime import datetime
import csv

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    LOG_DIR = config['paths'].get('logs', 'logs/')
except Exception:
    LOG_DIR = "logs/"  

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

class Logger:
    """Base class for different types of loggers."""
    def __init__(self, name, log_to_file=True):
        self.name = name
        self.log_to_file = log_to_file
        self.timers = {}
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if log_to_file:
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(f'logs/{name}_{timestamp}.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.scalars = {}
            
    def info(self, message):
        """Log an info message and print it to console if enabled"""
        message = message.replace('✓', '*')
        self.logger.info(message)
        if hasattr(self, 'console') and self.console:
            print(message)
        
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
        
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
        
    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
        
    def reset_timer(self, timer_name="default"):
        """Reset a timer with the given name."""
        self.timers[timer_name] = time.time()
        
    def log_elapsed_time(self, action_name, timer_name="default"):
        """Log the elapsed time since the timer was reset."""
        if timer_name not in self.timers:
            self.warning(f"Timer '{timer_name}' not initialized, can't log elapsed time.")
            return
            
        elapsed = time.time() - self.timers[timer_name]
        self.info(f"{action_name} completed in {elapsed:.2f} seconds.")
        return elapsed
        
    def add_scalar(self, name, value, step):
        """Add a scalar value for tracking (TensorBoard-like interface)."""
        if name not in self.scalars:
            self.scalars[name] = []
        self.scalars[name].append((step, value))
        
    def save_scalars(self, output_dir='logs/metrics'):
        """Save tracked scalars to CSV files."""
        if not self.scalars:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, values in self.scalars.items():
            safe_name = name.replace('/', '_')
            filename = f"{output_dir}/{safe_name}_{timestamp}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['step', 'value'])
                writer.writerows(values)
                
        self.info(f"Saved metrics to {output_dir}")

class TrainingLogger(Logger):
    """Logger specialized for training information."""
    def __init__(self):
        super().__init__("training")
        
    def training_progress(self, epoch, loss, val_loss, val_accuracy, val_value_mse=None, lr=None):
        """Log training progress metrics."""
        message = f"Epoch {epoch} - "
        message += f"Loss: {loss:.6f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%"
        
        if val_value_mse is not None:
            message += f", Val Value MSE: {val_value_mse:.6f}"
            
        if lr is not None:
            message += f", LR: {lr:.6f}"
            
        self.info(message)
        
        # Also add to TensorBoard-like tracking
        self.add_scalar('Loss/train', loss, epoch)
        self.add_scalar('Loss/val', val_loss, epoch)
        self.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        if val_value_mse is not None:
            self.add_scalar('ValueMSE/val', val_value_mse, epoch)
            
        if lr is not None:
            self.add_scalar('LearningRate', lr, epoch)
    
    def log_training_config(self, config):
        """Log the training configuration."""
        config_str = "Training Configuration:\n"
        for key, value in config.items():
            config_str += f"  {key}: {value}\n"
        self.info(config_str)

class EvaluationLogger(Logger):
    """Logger specialized for model evaluation."""
    def __init__(self):
        super().__init__("evaluation")
        
    def log_metrics(self, metrics):
        """Log evaluation metrics."""
        metrics_str = "Evaluation Metrics:\n"
        for key, value in metrics.items():
            metrics_str += f"  {key}: {value}\n"
            self.add_scalar(key, value, 0)  
        self.info(metrics_str)

class InferenceLogger(Logger):
    """Logger specialized for inference information."""
    def __init__(self):
        super().__init__("inference")
        
    def log_prediction(self, move, confidence, value=None):
        """Log a prediction."""
        message = f"Predicted move: {move}, Confidence: {confidence:.4f}"
        if value is not None:
            message += f", Value: {value:.4f}"
        self.info(message)
def get_training_logger():
    return TrainingLogger()

def get_evaluation_logger():
    return EvaluationLogger()

def get_inference_logger():
    return InferenceLogger()

if __name__ == "__main__":
    logger = get_training_logger()
    logger.info("Starting training...")
    
    for epoch in range(3):
        logger.training_progress(epoch+1, loss=0.456 - epoch*0.1, accuracy=75.5 + epoch*5)
    logger.log_metrics({"loss": 0.2345, "accuracy": 85.7, "f1_score": 0.876})
    logger.log_elapsed_time("Training") 