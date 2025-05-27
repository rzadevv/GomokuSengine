import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import numpy as np
import os
import queue
import argparse
import subprocess
import sys
from datetime import datetime
from PIL import Image, ImageTk  # For handling icons and images

# Set of colors for theming
COLORS = {
    "primary": "#4a6da7",     # Main blue color
    "secondary": "#7b8ab8",   # Lighter blue for accents
    "accent": "#f39c12",      # Orange for highlights/buttons
    "bg_light": "#f5f6fa",    # Light background
    "bg_dark": "#2c3e50",     # Dark background for contrast
    "text_light": "#ecf0f1",  # Light text color
    "text_dark": "#34495e",   # Dark text color
    "success": "#2ecc71",     # Green for success indicators
    "warning": "#e74c3c",     # Red for warnings/errors
}

class ThemedStyle(ttk.Style):
    """Custom style manager for the application."""
    
    def __init__(self):
        super().__init__()
        self._init_styles()
    
    def _init_styles(self):
        """Initialize custom styles for the application."""
        # Configure the basic theme
        self.theme_use('clam')  # 'clam' is a good base for customization
        
        # Configure TFrame
        self.configure('TFrame', background=COLORS["bg_light"])
        
        # Configure TLabelframe
        self.configure('TLabelframe', background=COLORS["bg_light"])
        self.configure('TLabelframe.Label', 
                      foreground=COLORS["primary"],
                      background=COLORS["bg_light"],
                      font=('Segoe UI', 10, 'bold'))
        
        # Configure TLabel
        self.configure('TLabel', 
                      background=COLORS["bg_light"],
                      foreground=COLORS["text_dark"])
        
        # Header label
        self.configure('Header.TLabel', 
                      font=('Segoe UI', 12, 'bold'),
                      foreground=COLORS["primary"])
        
        # Value display labels
        self.configure('Value.TLabel',
                      font=('Segoe UI', 10),
                      foreground=COLORS["text_dark"],
                      background=COLORS["bg_light"],
                      padding=5)
        
        # Metric value labels (for displaying actual metrics)
        self.configure('Metric.TLabel',
                      font=('Segoe UI', 10, 'bold'),
                      foreground=COLORS["primary"],
                      background=COLORS["bg_light"])
        
        # Configure TButton
        self.configure('TButton', 
                      background=COLORS["primary"],
                      foreground=COLORS["text_light"])
        
        # Configure primary action button
        self.configure('Action.TButton',
                      font=('Segoe UI', 10, 'bold'),
                      background=COLORS["accent"],
                      foreground=COLORS["text_light"])
        
        # Configure TEntry
        self.configure('TEntry', 
                      fieldbackground='white',
                      foreground=COLORS["text_dark"])
        
        # Configure Horizontal TProgressbar
        self.configure("Horizontal.TProgressbar", 
                      troughcolor=COLORS["bg_light"],
                      background=COLORS["accent"],
                      thickness=20)
        
        # Configure TCheckbutton
        self.configure('TCheckbutton',
                      background=COLORS["bg_light"],
                      foreground=COLORS["text_dark"])

class TrainingMonitor:
    """
    A thread-safe class to monitor training metrics.
    Acts as a buffer between the training process and the GUI.
    """
    def __init__(self, log_callback=None):
        self.metrics_queue = queue.Queue()
        self.raw_output_queue = queue.Queue()  # Store raw stdout for debugging
        self.running = False
        self.process = None
        self.log_callback = log_callback
        
    def start_training(self, cmd_args):
        """Start the training process with the given command-line arguments."""
        if self.running:
            return False
            
        try:
            # Log the command being executed
            cmd = ["python", "main.py", "train"] + cmd_args
            cmd_str = " ".join(cmd)
            if self.log_callback:
                self.log_callback(f"Executing command: {cmd_str}")
            
            # Create a script file with the command for better shell compatibility
            # This is especially important on Windows
            script_path = os.path.join(os.getcwd(), "run_training.py")
            with open(script_path, 'w') as f:
                f.write(f"""
import sys
import subprocess

# Execute the training command directly
cmd = {cmd}
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

# Forward all output to stdout/stderr so it can be captured
for line in process.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

for line in process.stderr:
    sys.stderr.write(line)
    sys.stderr.flush()

# Wait for process to complete
process.wait()
""")
            
            # Execute the script instead of the command directly
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True, 
                bufsize=1  # Line buffered
            )
            
            if self.log_callback:
                self.log_callback("Process started successfully")
            
            self.running = True
            
            # Start a thread to read the output
            self.reader_thread = threading.Thread(target=self._read_output)
            self.reader_thread.daemon = True
            self.reader_thread.start()
            
            return True
        except Exception as e:
            error_msg = f"Failed to start training: {str(e)}"
            if self.log_callback:
                self.log_callback(error_msg)
            messagebox.showerror("Error", error_msg)
            self.running = False
            return False
    
    def stop_training(self):
        """Stop the training process."""
        if not self.running:
            return
            
        self.running = False
        if self.process:
            if self.log_callback:
                self.log_callback("Stopping training process...")
            self.process.terminate()
            self.process.wait(timeout=5)  # Wait up to 5 seconds for process to terminate
            self.process = None
            if self.log_callback:
                self.log_callback("Training process terminated")
    
    def _read_output(self):
        """Read output from the training process and parse metrics."""
        epochs = []
        train_losses = []
        val_losses = []
        val_accuracies = []
        difficulty_thresholds = []
        last_epoch = 0
        
        if self.log_callback:
            self.log_callback("Starting to read process output...")
        
        while self.running and self.process:
            line = self.process.stdout.readline()
            if not line:
                # Check if process has ended
                if self.process.poll() is not None:
                    if self.log_callback:
                        self.log_callback(f"Process ended with return code: {self.process.returncode}")
                    self.running = False
                    break
                # Small delay to avoid CPU spinning
                time.sleep(0.1)
                continue
                
            # Store raw output for debugging
            if self.log_callback:
                self.log_callback(f"OUTPUT: {line.strip()}")
            
            # Queue the raw output for display
            self.raw_output_queue.put(line.strip())
            
            # Parse metrics from the output
            try:
                # Check for curriculum learning difficulty threshold
                if "Current difficulty threshold:" in line:
                    try:
                        threshold_str = line.strip().split("Current difficulty threshold:")[1].strip()
                        threshold = float(threshold_str)
                        if last_epoch > 0:  # Only add if we have seen an epoch
                            difficulty_thresholds.append(threshold)
                            self.metrics_queue.put({"difficulty_threshold": threshold, "diff_thresholds": difficulty_thresholds.copy()})
                            if self.log_callback:
                                self.log_callback(f"Parsed difficulty threshold: {threshold}")
                    except Exception as e:
                        if self.log_callback:
                            self.log_callback(f"Error parsing difficulty threshold: {str(e)}")

                # Look for lines with epoch info in the format we expect from training
                if "|" in line:
                    if self.log_callback:
                        self.log_callback(f"Found metrics line: {line.strip()}")
                    # Don't process the header line
                    if "Epoch |" in line or "------" in line:
                        continue
                        
                    parts = line.strip().split("|")
                    if len(parts) >= 6:  # We have multiple columns with validation metrics
                        try:
                            # Parse each metric from the line
                            epoch_str = parts[0].strip()
                            train_loss_str = parts[1].strip()
                            val_loss_str = parts[2].strip()
                            val_acc_str = parts[3].strip()
                            improvement_str = parts[4].strip()
                            
                            if self.log_callback:
                                self.log_callback(f"Parsing: {epoch_str=}, {train_loss_str=}, {val_loss_str=}, {val_acc_str=}")
                            
                            # Convert to appropriate types
                            epoch = int(epoch_str)
                            train_loss = float(train_loss_str)
                            val_loss = float(val_loss_str)
                            # Clean the val_acc string to handle "%" if present
                            val_acc = float(val_acc_str.rstrip('%'))
                            
                            if epoch > last_epoch:
                                epochs.append(epoch)
                                train_losses.append(train_loss)
                                val_losses.append(val_loss)
                                val_accuracies.append(val_acc)
                                last_epoch = epoch
                                
                                # Put the metrics in the queue for the GUI to consume
                                metrics_data = {
                                    "epoch": epoch,
                                    "train_loss": train_loss,
                                    "val_loss": val_loss,
                                    "val_accuracy": val_acc,
                                    "improvement": improvement_str,
                                    "epochs": epochs.copy(),
                                    "train_losses": train_losses.copy(),
                                    "val_losses": val_losses.copy(),
                                    "val_accuracies": val_accuracies.copy(),
                                    "diff_thresholds": difficulty_thresholds.copy() if difficulty_thresholds else None
                                }
                                self.metrics_queue.put(metrics_data)
                                
                                if self.log_callback:
                                    self.log_callback(f"Successfully parsed metrics for epoch {epoch}")
                        except (ValueError, IndexError) as e:
                            if self.log_callback:
                                self.log_callback(f"Error parsing metrics: {str(e)}")
            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"Unexpected error: {str(e)}")
                
            # Also check for errors
            err_line = self.process.stderr.readline()
            if err_line:
                if self.log_callback:
                    self.log_callback(f"ERROR: {err_line.strip()}")
                self.metrics_queue.put({"error": err_line.strip()})
                self.raw_output_queue.put(f"ERROR: {err_line.strip()}")
                
        # Process has ended
        self.metrics_queue.put({"finished": True})
        if self.log_callback:
            self.log_callback("Output reader thread completed")
            
    def get_metrics(self):
        """Get the latest metrics from the queue (non-blocking)."""
        if self.metrics_queue.empty():
            return None
        
        # Get all available metrics
        metrics = None
        while not self.metrics_queue.empty():
            metrics = self.metrics_queue.get()
            
        return metrics
        
    def get_raw_output(self):
        """Get raw output for debugging (non-blocking)."""
        output_lines = []
        while not self.raw_output_queue.empty():
            output_lines.append(self.raw_output_queue.get())
        
        return output_lines if output_lines else None

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku Training Monitor")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        
        # Apply custom styles
        self.style = ThemedStyle()
        
        # Configure the root window background
        self.root.configure(background=COLORS["bg_light"])
        
        # Make root window responsive to resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)  # Header row doesn't need to resize
        self.root.rowconfigure(1, weight=1)  # Main content should resize
        self.root.rowconfigure(2, weight=0)  # Status bar doesn't need to resize
        
        # Create the monitor
        self.monitor = TrainingMonitor(log_callback=self.log_message)
        
        # Create header with title and logo
        self._create_header(root)
        
        # Set up the main frame with decent padding
        main_frame = ttk.Frame(root, padding="20", style='TFrame')
        main_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(10, 15))
        
        # Configure grid columns and rows to be resizable
        main_frame.columnconfigure(0, weight=2)  # Configuration column
        main_frame.columnconfigure(1, weight=3)  # Results column
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left side - Training Configuration with rounded corners and shadow
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding="15")
        config_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 15))
        
        # Create the configuration panels using grid layout for better resizing
        self._create_config_panels(config_frame)
        
        # Right side top - Current Metrics panel with visual emphasis
        metrics_frame = ttk.LabelFrame(main_frame, text="Current Metrics", padding="15")
        metrics_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self._create_metrics_display(metrics_frame)
        
        # Right side bottom - Training Log with improved styling
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="15")
        log_frame.grid(row=1, column=1, sticky="nsew")
        self._create_log_panel(log_frame)
        
        # Status bar
        self._create_status_bar(root)
        
        # Set up the update timer
        self._setup_update_timer()
        
        # Log initialization
        self.log_message("💻 Gomoku Training Monitor initialized and ready")
        
    def _create_header(self, parent):
        """Create an attractive header for the application."""
        header_frame = ttk.Frame(parent, style='TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        
        # Configure header frame for proper resizing
        header_frame.columnconfigure(0, weight=1)  # Title column expands
        header_frame.columnconfigure(1, weight=0)  # Icon column fixed size
        
        # Add title label
        title_label = ttk.Label(header_frame, 
                                text="Gomoku AI Training Dashboard", 
                                font=('Segoe UI', 16, 'bold'),
                                foreground=COLORS["primary"],
                                background=COLORS["bg_light"])
        title_label.grid(row=0, column=0, sticky="w")
        
        # Try to load and display a small Gomoku icon if available
        try:
            # You would need to create/add this icon file to your project
            icon_path = os.path.join(os.path.dirname(__file__), "gomoku_icon.png")
            if os.path.exists(icon_path):
                img = Image.open(icon_path)
                img = img.resize((32, 32), Image.LANCZOS)
                self.icon_img = ImageTk.PhotoImage(img)
                icon_label = ttk.Label(header_frame, image=self.icon_img, background=COLORS["bg_light"])
                icon_label.grid(row=0, column=1, padx=10)
        except Exception as e:
            # If icon loading fails, just display a text logo
            logo_label = ttk.Label(header_frame, 
                                  text="⚫ ⚪",
                                  font=('Segoe UI', 14, 'bold'),
                                  foreground=COLORS["primary"],
                                  background=COLORS["bg_light"])
            logo_label.grid(row=0, column=1, padx=10)
    
    def _create_config_panels(self, parent):
        """Create configuration panels using grid layout for better resizing."""
        # Configure parent for proper grid layout
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)  # Dataset selection
        parent.rowconfigure(1, weight=0)  # Training options
        parent.rowconfigure(2, weight=0)  # Advanced options
        parent.rowconfigure(3, weight=0)  # Buttons
        parent.rowconfigure(4, weight=1)  # Empty space that can expand
        
        # Dataset selection panel
        dataset_frame = ttk.Frame(parent, padding="10")
        dataset_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self._create_dataset_panel(dataset_frame)
        
        # Add a separator
        ttk.Separator(parent, orient='horizontal').grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Training options panel
        training_frame = ttk.Frame(parent, padding="10")
        training_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self._create_training_options(training_frame)
        
        # Add a separator
        ttk.Separator(parent, orient='horizontal').grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Advanced options panel
        advanced_frame = ttk.Frame(parent, padding="10")
        advanced_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self._create_advanced_options(advanced_frame)
        
        # Add a separator
        ttk.Separator(parent, orient='horizontal').grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        
        # Buttons panel
        button_frame = ttk.Frame(parent, padding="10")
        button_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        self._create_button_panel(button_frame)
    
    def _create_dataset_panel(self, parent):
        """Create the dataset selection panel."""
        # Ensure the frame expands properly
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=0)
        
        # Section title
        ttk.Label(parent, text="Dataset Selection", style='Header.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 8))
        
        # Dataset selection with browse button
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=0)
        
        self.dataset_var = tk.StringVar(value="gomoku_dataset.npz")
        dataset_entry = ttk.Entry(input_frame, textvariable=self.dataset_var)
        dataset_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        dataset_button = ttk.Button(input_frame, text="Browse", command=self._browse_dataset)
        dataset_button.grid(row=0, column=1, sticky="e")

    def _create_training_options(self, parent):
        """Create the main training parameters panel using grid layout."""
        # Configure for grid layout
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)  # Header
        parent.rowconfigure(1, weight=0)  # Parameters grid
        parent.rowconfigure(2, weight=0)  # Save path
        parent.rowconfigure(3, weight=0)  # Stable LR checkbox
        
        # Section title
        ttk.Label(parent, text="Training Parameters", style='Header.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Create parameter grid with 2 columns
        params_grid = ttk.Frame(parent)
        params_grid.grid(row=1, column=0, sticky="ew")
        params_grid.columnconfigure(0, weight=1)
        params_grid.columnconfigure(1, weight=1)
        
        # Row 0: Epochs and Learning Rate
        self._create_param_entry(params_grid, "Epochs:", "epochs_var", 20, 0, 0)
        self._create_param_entry(params_grid, "Learning Rate:", "lr_var", 0.001, 0, 1)
        
        # Row 1: Batch Size and Label Smoothing
        self._create_param_entry(params_grid, "Batch Size:", "batch_size_var", 64, 1, 0)
        self._create_param_entry(params_grid, "Label Smoothing:", "smoothing_var", 0.1, 1, 1)
        
        # Row 2: Policy Weight and Value Weight
        self._create_param_entry(params_grid, "Policy Weight:", "policy_weight_var", 1.0, 2, 0)
        self._create_param_entry(params_grid, "Value Weight:", "value_weight_var", 1.3, 2, 1)
        
        # Row 3: Model save path
        save_frame = ttk.Frame(parent)
        save_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        save_frame.columnconfigure(0, weight=1)
        
        ttk.Label(save_frame, text="Model Save Path:").grid(row=0, column=0, sticky="w", pady=(5, 5))
        self.model_path_var = tk.StringVar(value="best_gomoku_model.pth")
        ttk.Entry(save_frame, textvariable=self.model_path_var).grid(row=1, column=0, sticky="ew")
        
        # Stable learning rate checkbox
        self.stable_lr_var = tk.BooleanVar(value=False)
        stable_lr_check = ttk.Checkbutton(parent, 
                                        text="Use Stable Learning Rate", 
                                        variable=self.stable_lr_var)
        stable_lr_check.grid(row=3, column=0, sticky="w", pady=(10, 0))
    
    def _create_advanced_options(self, parent):
        """Create the advanced options panel using grid layout."""
        # Configure for grid layout
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)  # Header
        parent.rowconfigure(1, weight=0)  # Options grid
        parent.rowconfigure(2, weight=0)  # Curriculum frame
        
        # Section title
        ttk.Label(parent, text="Advanced Options", style='Header.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Create left and right columns for options
        options_grid = ttk.Frame(parent)
        options_grid.grid(row=1, column=0, sticky="ew")
        options_grid.columnconfigure(0, weight=1)
        options_grid.columnconfigure(1, weight=1)
        
        # Left column - Distributed training
        dist_frame = ttk.LabelFrame(options_grid, text="Distributed Training", padding=10)
        dist_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        dist_frame.columnconfigure(0, weight=1)
        
        self.distributed_var = tk.BooleanVar(value=False)
        dist_check = ttk.Checkbutton(dist_frame, text="Use Multiple GPUs", variable=self.distributed_var)
        dist_check.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        gpu_frame = ttk.Frame(dist_frame)
        gpu_frame.grid(row=1, column=0, sticky="ew")
        
        ttk.Label(gpu_frame, text="Number of GPUs:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.num_gpus_var = tk.IntVar(value=2)
        ttk.Entry(gpu_frame, textvariable=self.num_gpus_var, width=5).grid(row=0, column=1, sticky="w")
        
        # Right column - Validation settings
        val_frame = ttk.LabelFrame(options_grid, text="Validation Settings", padding=10)
        val_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        val_frame.columnconfigure(0, weight=1)
        
        val_split_frame = ttk.Frame(val_frame)
        val_split_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(val_split_frame, text="Validation Split:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.val_split_var = tk.DoubleVar(value=0.1)
        ttk.Entry(val_split_frame, textvariable=self.val_split_var, width=5).grid(row=0, column=1, sticky="w")
        
        patience_frame = ttk.Frame(val_frame)
        patience_frame.grid(row=1, column=0, sticky="ew")
        
        ttk.Label(patience_frame, text="Early Stopping Patience:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.patience_var = tk.IntVar(value=5)
        ttk.Entry(patience_frame, textvariable=self.patience_var, width=5).grid(row=0, column=1, sticky="w")
        
        # Curriculum learning frame (below the grid)
        curr_frame = ttk.LabelFrame(parent, text="Curriculum Learning", padding=10)
        curr_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        curr_frame.columnconfigure(0, weight=1)
        
        self.curriculum_var = tk.BooleanVar(value=False)
        curr_check = ttk.Checkbutton(curr_frame, text="Enable Curriculum Learning", variable=self.curriculum_var)
        curr_check.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        epochs_frame = ttk.Frame(curr_frame)
        epochs_frame.grid(row=1, column=0, sticky="ew")
        
        ttk.Label(epochs_frame, text="Curriculum Epochs:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.curriculum_epochs_var = tk.IntVar(value=15)
        ttk.Entry(epochs_frame, textvariable=self.curriculum_epochs_var, width=5).grid(row=0, column=1, sticky="w")

    def _create_button_panel(self, parent):
        """Create the action buttons panel."""
        # Configure for proper grid layout
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        
        self.start_button = ttk.Button(parent, 
                                    text="▶ Start Training", 
                                    command=self._start_training,
                                    style='Action.TButton')
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5), ipady=5)
        
        self.stop_button = ttk.Button(parent, 
                                    text="■ Stop Training", 
                                    command=self._stop_training, 
                                    state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0), ipady=5)

    def _create_metrics_display(self, parent):
        """Create an attractive metrics display with visuals."""
        # Create a stylish container for metrics
        metrics_container = ttk.Frame(parent)
        metrics_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure container for proper resizing
        metrics_container.columnconfigure(0, weight=1)
        metrics_container.rowconfigure(0, weight=1)  # Metrics row
        metrics_container.rowconfigure(1, weight=0)  # Improvement row
        metrics_container.rowconfigure(2, weight=0)  # Progress row
        
        # Add row of metric cards
        metrics_row = ttk.Frame(metrics_container)
        metrics_row.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Configure equal-width columns
        for i in range(4):
            metrics_row.columnconfigure(i, weight=1)
        
        # Create metric cards with visual indicators
        self._create_metric_card(metrics_row, "Current Epoch", "epoch_label", "--", 0)
        self._create_metric_card(metrics_row, "Training Loss", "train_loss_label", "--", 1)
        self._create_metric_card(metrics_row, "Validation Loss", "val_loss_label", "--", 2)
        self._create_metric_card(metrics_row, "Accuracy", "val_accuracy_label", "--", 3)
        
        # Add improvement indicator separately
        imp_frame = ttk.Frame(metrics_container, padding=(10, 5))
        imp_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(imp_frame, text="Improvement Status:", style='Value.TLabel').grid(row=0, column=0, sticky="w")
        self.improvement_label = ttk.Label(imp_frame, text="--", style='Metric.TLabel')
        self.improvement_label.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        # Add an attractive progress bar with label
        progress_frame = ttk.Frame(metrics_container, padding=(10, 5))
        progress_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        ttk.Label(progress_frame, text="Training Progress:", style='Header.TLabel').grid(row=0, column=0, sticky="w")
        
        # Progress bar with percentage indicator
        progress_container = ttk.Frame(progress_frame, padding=(0, 5))
        progress_container.grid(row=1, column=0, sticky="ew")
        progress_container.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_container, 
            variable=self.progress_var, 
            maximum=100,
            style="Horizontal.TProgressbar"
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        
        # Percentage label that will update with the progress
        self.progress_percent = ttk.Label(
            progress_container, 
            text="0%", 
            style='Metric.TLabel',
            background=COLORS["bg_light"]
        )
        self.progress_percent.grid(row=0, column=1, sticky="w", padx=(5, 0))  # Place beside progress bar
    
    def _create_metric_card(self, parent, title, label_var_name, default_value, column):
        """Create a stylish card-like display for a single metric."""
        card_frame = ttk.Frame(parent, padding=10)
        card_frame.grid(row=0, column=column, sticky="nsew", padx=5)
        
        # Add an internal border and background to simulate a card
        card_inside = ttk.Frame(card_frame, padding=8)
        card_inside.grid(row=0, column=0, sticky="nsew")
        card_frame.columnconfigure(0, weight=1)
        card_frame.rowconfigure(0, weight=1)
        card_inside.configure(style='TFrame')
        
        # Configure for proper resizing
        card_inside.columnconfigure(0, weight=1)
        card_inside.rowconfigure(0, weight=0)  # Title
        card_inside.rowconfigure(1, weight=1)  # Value
        
        # Metric title
        ttk.Label(card_inside, text=title, style='Value.TLabel').grid(row=0, column=0, sticky="n")
        
        # Metric value display
        setattr(self, label_var_name, ttk.Label(
            card_inside, 
            text=default_value, 
            font=('Segoe UI', 13, 'bold'),
            foreground=COLORS["primary"],
            background=COLORS["bg_light"],
            anchor="center"
        ))
        getattr(self, label_var_name).grid(row=1, column=0, sticky="n", pady=5)
    
    def _create_log_panel(self, parent):
        """Create an improved log panel with syntax highlighting."""
        # Create a container for the log
        log_container = ttk.Frame(parent)
        log_container.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Configure container for proper resizing
        log_container.columnconfigure(0, weight=1)
        log_container.rowconfigure(0, weight=1)
        
        # Create a scrolled text widget with custom styling
        self.log_text = scrolledtext.ScrolledText(
            log_container,
            wrap=tk.WORD,
            font=('Consolas', 9),
            background="#f8f9fa",
            foreground="#333333",
            borderwidth=1,
            relief="solid"
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Configure text tags for different message types
        self.log_text.tag_configure("timestamp", foreground="#7f8c8d")
        self.log_text.tag_configure("info", foreground="#2980b9")
        self.log_text.tag_configure("success", foreground="#27ae60")
        self.log_text.tag_configure("warning", foreground="#d35400")
        self.log_text.tag_configure("error", foreground="#c0392b")
        self.log_text.tag_configure("output", foreground="#2c3e50")
        self.log_text.tag_configure("bold", font=('Consolas', 9, 'bold'))
    
    def _create_status_bar(self, parent):
        """Create an improved status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=0, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        
        # Configure frame for proper resizing
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=0)  # Separator
        status_frame.rowconfigure(1, weight=0)  # Status container
        
        # Add a thin separator above the status bar
        ttk.Separator(status_frame, orient='horizontal').grid(row=0, column=0, sticky="ew")
        
        # Status text with icon indicator
        status_container = ttk.Frame(status_frame, padding=(10, 5))
        status_container.grid(row=1, column=0, sticky="ew")
        
        # Configure status container for proper resizing
        status_container.columnconfigure(1, weight=1)  # Status text should expand
        
        # Status indicator (colored circle)
        self.status_indicator = ttk.Label(
            status_container,
            text="⬤",  # Unicode circle character
            foreground=COLORS["success"],
            background=COLORS["bg_light"]
        )
        self.status_indicator.grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        # Status text
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            status_container, 
            textvariable=self.status_var,
            background=COLORS["bg_light"]
        )
        status_label.grid(row=0, column=1, sticky="ew")
        
        # Add timestamp on the right side
        self.timestamp_var = tk.StringVar(value=datetime.now().strftime("%H:%M:%S"))
        timestamp_label = ttk.Label(
            status_container,
            textvariable=self.timestamp_var,
            foreground=COLORS["secondary"],
            background=COLORS["bg_light"]
        )
        timestamp_label.grid(row=0, column=2, sticky="e")
        
        # Update the timestamp periodically
        def update_timestamp():
            self.timestamp_var.set(datetime.now().strftime("%H:%M:%S"))
            parent.after(1000, update_timestamp)
            
        parent.after(1000, update_timestamp)
        
    def log_message(self, message):
        """Add a message to the log with timestamp and formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # This needs to be thread-safe since it might be called from different threads
        def _log():
            # Insert timestamp with tag
            self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            
            # Determine message type and apply appropriate tag
            if message.startswith("Error") or "ERROR" in message:
                self.log_text.insert(tk.END, message + "\n", "error")
            elif message.startswith(">>"):
                self.log_text.insert(tk.END, message + "\n", "output")
            elif "success" in message.lower() or "completed" in message.lower():
                self.log_text.insert(tk.END, message + "\n", "success")
            elif "warning" in message.lower():
                self.log_text.insert(tk.END, message + "\n", "warning")
            else:
                self.log_text.insert(tk.END, message + "\n", "info")
                
            # Auto-scroll to the end
            self.log_text.see(tk.END)
        
        # If called from a non-main thread, use after() to safely update UI
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, _log)
        else:
            _log()
        
    def _browse_dataset(self):
        """Open a file dialog to select the dataset file."""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("NumPy Files", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.dataset_var.set(filename)
            self.log_message(f"Selected dataset: {filename}")
    
    def _start_training(self):
        """Start training with the current parameters."""
        if not self.monitor or self.monitor.running:
            return
            
        # Validate inputs
        try:
            dataset_path = self.dataset_var.get().strip()
            if not os.path.exists(dataset_path):
                messagebox.showerror("Error", f"Dataset file not found: {dataset_path}")
                return
                
            # Update status indicator
            self.status_indicator.config(foreground=COLORS["accent"])
            self.status_var.set("Initializing training...")
                
            # Build command-line arguments
            cmd_args = [
                # Dataset path
                "--npz", dataset_path,
                
                # Training parameters
                "--epochs", str(self.epochs_var.get()),
                "--batch_size", str(self.batch_size_var.get()),
                "--lr", str(self.lr_var.get()),
                "--smoothing", str(self.smoothing_var.get()),
                "--policy_weight", str(self.policy_weight_var.get()),
                "--value_weight", str(self.value_weight_var.get()),
                
                # Output path
                "--model_path", self.model_path_var.get(),
                
                # Validation settings
                "--val_split", str(self.val_split_var.get()),
                "--patience", str(self.patience_var.get())
            ]
            
            # Add stable learning rate flag if enabled
            if self.stable_lr_var.get():
                cmd_args.append("--stable_lr")
            
            # Distributed training options
            if self.distributed_var.get():
                cmd_args.append("--distributed")
                
                # Add number of GPUs if specified
                if self.num_gpus_var.get() > 0:
                    cmd_args.extend(["--num_gpus", str(self.num_gpus_var.get())])
            
            # Curriculum learning options
            if self.curriculum_var.get():
                cmd_args.append("--curriculum")
                cmd_args.extend(["--curriculum_epochs", str(self.curriculum_epochs_var.get())])
            
            # Create a fresh TrainingMonitor for this run
            self.monitor = TrainingMonitor(log_callback=self.log_message)
            
            # Reset UI elements
            self._reset_ui()
            
            # Start training with animation
            self.start_button.config(state=tk.DISABLED)
            
            # Use after() to provide visual feedback before starting the process
            def delayed_start():
                success = self.monitor.start_training(cmd_args)
                if success:
                    self.log_message("✅ Training started successfully")
                    self.status_indicator.config(foreground=COLORS["success"])
                    self.status_var.set("Training in progress")
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                    
                    # Start the UI update timer
                    self._setup_update_timer()
                else:
                    self.log_message("❌ Failed to start training")
                    self.status_indicator.config(foreground=COLORS["warning"])
                    self.status_var.set("Training failed to start")
                    self.start_button.config(state=tk.NORMAL)
            
            self.root.after(200, delayed_start)
            
        except Exception as e:
            self.log_message(f"❌ Error starting training: {str(e)}")
            self.status_indicator.config(foreground=COLORS["warning"])
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error starting training: {str(e)}")
            self.start_button.config(state=tk.NORMAL)
    
    def _stop_training(self):
        """Stop the training process with visual feedback."""
        if not self.monitor.running:
            self.log_message("⚠️ No training in progress to stop")
            return
            
        # Update UI to show stopping state    
        self.status_indicator.config(foreground=COLORS["warning"])
        self.status_var.set("Stopping training...")
        self.stop_button.config(state=tk.DISABLED)
        
        # Use after() to provide visual feedback before actually stopping
        def delayed_stop():
            self.log_message("⏹️ Stopping training process...")
            self.monitor.stop_training()
            self.status_indicator.config(foreground=COLORS["accent"])
            self.status_var.set("Training stopped by user")
            self.start_button.config(state="normal")
        
        self.root.after(200, delayed_stop)
    
    def _reset_ui(self):
        """Reset the UI elements to their initial state with visual feedback."""
        # Reset metrics
        self.epoch_label.config(text="--")
        self.train_loss_label.config(text="--")
        self.val_loss_label.config(text="--")
        self.val_accuracy_label.config(text="--")
        self.improvement_label.config(text="--")
        
        # Reset progress bar
        self.progress_var.set(0)
        self.progress_percent.config(text="0%")
        
        # Clear the log (but keep initialization message)
        self.log_text.delete(1.0, tk.END)
        self.log_message("🔄 UI reset completed")
    
    def _setup_update_timer(self):
        """Set up a timer to periodically update the UI with metrics."""
        def update_ui():
            # Process raw output
            raw_output = self.monitor.get_raw_output()
            if raw_output:
                for line in raw_output:
                    if line.strip():  # Skip empty lines
                        self.log_message(f">> {line}")
                        
            # Process metrics updates
            if self.monitor and self.monitor.running:
                # Get the latest metrics
                metrics = self.monitor.get_metrics()
                
                if metrics:
                    if "error" in metrics:
                        error_msg = f"Error: {metrics['error']}"
                        self.status_indicator.config(foreground=COLORS["warning"])
                        self.status_var.set(error_msg)
                        self.log_message(f"❌ {error_msg}")
                    elif "finished" in metrics:
                        self.status_indicator.config(foreground=COLORS["success"])
                        self.status_var.set("Training finished successfully")
                        self.start_button.config(state="normal")
                        self.stop_button.config(state="disabled")
                        self.monitor.running = False
                        self.log_message("✅ Training process completed")
                    elif "difficulty_threshold" in metrics:
                        # Just update the log with the new threshold
                        threshold = metrics["difficulty_threshold"]
                        self.log_message(f"🔄 Difficulty threshold updated to {threshold:.3f}")
                    else:
                        # Update metrics display with visual feedback
                        epoch = metrics["epoch"]
                        train_loss = metrics["train_loss"]
                        val_loss = metrics["val_loss"]
                        val_accuracy = metrics["val_accuracy"]
                        improvement = metrics["improvement"]
                        
                        # Update display labels with formatted values
                        self.epoch_label.config(text=f"{epoch}")
                        self.train_loss_label.config(text=f"{train_loss:.6f}")
                        self.val_loss_label.config(text=f"{val_loss:.6f}")
                        self.val_accuracy_label.config(text=f"{val_accuracy:.2f}%")
                        
                        # Style the improvement text based on content
                        improvement_text = improvement
                        if "improved" in improvement.lower():
                            improvement_text = f"✓ {improvement}"
                            self.improvement_label.config(
                                text=improvement_text,
                                foreground=COLORS["success"]
                            )
                        else:
                            improvement_text = f"× {improvement}"
                            self.improvement_label.config(
                                text=improvement_text,
                                foreground=COLORS["text_dark"]
                            )
                        
                        # Update progress bar and percentage
                        progress = (epoch / self.epochs_var.get()) * 100
                        self.progress_var.set(progress)
                        self.progress_percent.config(text=f"{progress:.1f}%")
                        
                        # Update status bar
                        self.status_indicator.config(foreground=COLORS["success"])
                        self.status_var.set(f"Training in progress - Epoch {epoch}/{self.epochs_var.get()}")
                        
                        # Log the update but not for every epoch to avoid clutter
                        if epoch % 5 == 0 or epoch == 1:
                            self.log_message(f"📊 Updated metrics for epoch {epoch}")
            
            # Schedule the next update
            self.root.after(1000, update_ui)
            
        # Start the timer
        self.root.after(1000, update_ui)

    def _create_param_entry(self, parent, label_text, var_name, default_value, row, col):
        """Helper to create a consistently styled parameter entry."""
        frame = ttk.Frame(parent, padding=(5, 5))
        frame.grid(row=row, column=col, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(0, weight=1)  # Make sure entry expands properly
        
        ttk.Label(frame, text=label_text).grid(row=0, column=0, sticky="w", pady=(0, 3))
        
        # Create the appropriate variable based on default value type
        if isinstance(default_value, bool):
            setattr(self, var_name, tk.BooleanVar(value=default_value))
        elif isinstance(default_value, int):
            setattr(self, var_name, tk.IntVar(value=default_value))
        elif isinstance(default_value, float):
            setattr(self, var_name, tk.DoubleVar(value=default_value))
        else:
            setattr(self, var_name, tk.StringVar(value=str(default_value)))
            
        entry = ttk.Entry(frame, textvariable=getattr(self, var_name), width=12)
        entry.grid(row=1, column=0, sticky="ew")

def main():
    parser = argparse.ArgumentParser(description="Gomoku Training GUI")
    parser.add_argument("--dataset", type=str, default="gomoku_dataset.npz", 
                       help="Path to the dataset file to pre-populate in the GUI")
    args = parser.parse_args()
    
    root = tk.Tk()
    # Try to set a window icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "gomoku_icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass  # Silently ignore if icon setting fails
        
    app = TrainingGUI(root)
    
    # Set initial dataset if provided
    if args.dataset:
        app.dataset_var.set(args.dataset)
    
    root.mainloop()

if __name__ == "__main__":
    main() 