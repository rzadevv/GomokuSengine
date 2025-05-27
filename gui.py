import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas, Frame, Label, Button, Scale, StringVar, DoubleVar, BooleanVar, IntVar
import os
import numpy as np
import torch
import PIL.Image
import PIL.ImageTk
from inference import load_model, predict_move_with_details, predict_move_with_weighted_details

# Constants
BOARD_SIZE = 15
CELL_SIZE = 40
STONE_RADIUS = 18
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE
MODEL_PATH = "best_gomoku_model.pth"

# Theme colors - Updated with better visuals
BOARD_COLOR = "#E9BC5C"  
BLACK = "#000000"
WHITE = "#FFFFFF"
RED = "#FF0000"
BLUE = "#0000FF"
GREEN = "#008800"
LIGHT_GRAY = "#F0F0F0"
DARK_GRAY = "#606060"
HIGHLIGHT_COLOR = "#4CAF50"  # Material design green
ACCENT_COLOR = "#2196F3"     # Material design blue
SHADOW_COLOR = "#555555"     # Shadow color for stones
LAST_MOVE_COLOR = "#FF4444"  # Brighter red for last move indicator
STAR_POINT_COLOR = "#000000" # Color for star points

class GomokuBoard(Canvas):
    """Canvas widget for drawing the Gomoku board and stones"""
    def __init__(self, parent, cell_size=CELL_SIZE, board_size=BOARD_SIZE, **kwargs):
        self.cell_size = cell_size
        self.board_size = board_size
        canvas_width = board_size * cell_size
        canvas_height = board_size * cell_size
        
        # Add board_array attribute
        self.board_array = None
        
        super().__init__(parent, width=canvas_width, height=canvas_height, 
                         highlightthickness=1, **kwargs)
        
        # Try to load a board texture image
        try:
            self.board_image = PIL.Image.open("board_texture.jpg").resize(
                (canvas_width, canvas_height), PIL.Image.LANCZOS)
            self.board_photo = PIL.ImageTk.PhotoImage(self.board_image)
            self.has_texture = True
        except:
            self.has_texture = False
        
        self.draw_board_grid()
    
    def draw_board_grid(self):
        """Draw the board grid lines and star points"""
        # Draw board background - use a more realistic wood texture color
        self.create_rectangle(
            0, 0, 
            self.board_size * self.cell_size,
            self.board_size * self.cell_size,
            fill=BOARD_COLOR, outline=''
        )
        
        # Draw grid lines - make them darker and thinner for better contrast
        for i in range(self.board_size):
            # Horizontal lines
            self.create_line(
                self.cell_size // 2, 
                i * self.cell_size + self.cell_size // 2,
                self.board_size * self.cell_size - self.cell_size // 2, 
                i * self.cell_size + self.cell_size // 2,
                width=1, fill="#000000"
            )
            # Vertical lines
            self.create_line(
                i * self.cell_size + self.cell_size // 2, 
                self.cell_size // 2,
                i * self.cell_size + self.cell_size // 2, 
                self.board_size * self.cell_size - self.cell_size // 2,
                width=1, fill="#000000"
            )
        
        # Draw star points (dots at specific intersections)
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for x, y in star_points:
            self.create_oval(
                x * self.cell_size + self.cell_size // 2 - 4,
                y * self.cell_size + self.cell_size // 2 - 4,
                x * self.cell_size + self.cell_size // 2 + 4,
                y * self.cell_size + self.cell_size // 2 + 4,
                fill="#000000", outline=""
            )
    
    def draw_stone(self, row, col, is_black, is_last_move=False):
        """Draw a stone at the specified position"""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        radius = STONE_RADIUS
        
        # Draw shadow for 3D effect - more pronounced
        shadow_offset = 3
        self.create_oval(
            x - radius + shadow_offset,
            y - radius + shadow_offset,
            x + radius + shadow_offset,
            y + radius + shadow_offset,
            fill="#333333", outline='', tags="stone"
        )
        
        # Draw the stone with better 3D effect
        if is_black:
            # Black stone - solid with no outline
            self.create_oval(x-radius, y-radius, x+radius, y+radius, 
                            fill="#000000", outline='', tags="stone")
            
            # Add subtle highlight to give 3D appearance
            highlight_size = radius * 0.4
            highlight_offset = radius * 0.3
            self.create_oval(
                x-highlight_size-highlight_offset, 
                y-highlight_size-highlight_offset,
                x+highlight_size-highlight_offset, 
                y+highlight_size-highlight_offset,
                fill='#444444', outline='', tags="stone"
            )
        else:
            # White stone - with thin black outline
            self.create_oval(x-radius, y-radius, x+radius, y+radius, 
                            fill="#FFFFFF", outline='#000000', width=1, tags="stone")
            
            # Add subtle gradient for 3D effect
            highlight_size = radius * 0.6
            self.create_oval(
                x-highlight_size, 
                y-highlight_size,
                x+highlight_size, 
                y+highlight_size,
                fill='', outline='#DDDDDD', tags="stone"
            )
        
        # Mark last move with a small red dot - make it more visible
        if is_last_move:
            self.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill="#FF4444", outline='', tags="last_move"
            )
    
    def draw_probability_heatmap(self, probabilities, board_array=None, max_prob=None):
        """Draw a heatmap showing the model's move predictions"""
        if max_prob is None:
            max_prob = np.max(probabilities) if np.max(probabilities) > 0 else 1.0
        
        # Use provided board_array or instance board_array
        board_array = board_array if board_array is not None else self.board_array
        if board_array is None:
            return  # No board state available
            
        # Clear existing heatmap indicators
        self.delete("heatmap")
        
        # Create a list of colors to use for different probability levels
        heat_colors = ['#FFE5E5', '#FFCCCC', '#FFB3B3', '#FF9999', '#FF8080']
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                prob = probabilities[row, col]
                if prob > 0.01:  # Only draw if probability is significant
                    x = col * self.cell_size + self.cell_size // 2
                    y = row * self.cell_size + self.cell_size // 2
                    
                    # Scale radius based on probability
                    radius = int(STONE_RADIUS * 0.6 * (prob / max_prob))
                    
                    # Choose color based on probability
                    color_idx = min(int(prob / max_prob * len(heat_colors)), len(heat_colors) - 1)
                    color = heat_colors[color_idx]
                    
                    # Use rectangles for cleaner heatmap effect
                    if board_array[row, col, 2] == 1.0:  # Only if position is empty
                        self.create_oval(
                            x - radius, y - radius,
                            x + radius, y + radius,
                            fill=color, outline='',
                            tags="heatmap"
                        )
                        
                        # For highest probability, add small red dot in center
                        if prob > 0.8 * max_prob:
                            self.create_oval(
                                x - 2, y - 2,
                                x + 2, y + 2,
                                fill="#CC0000", outline='',
                                tags="heatmap"
                            )
    
    def clear_board(self):
        """Clear all stones and markers from the board"""
        self.delete("all")
        self.draw_board_grid()

class InfoPanel(ttk.Frame):  # Use ttk.Frame for better theming
    """Panel for displaying game information and controls"""
    def __init__(self, parent, gui):
        super().__init__(parent, width=INFO_PANEL_WIDTH, padding="10 10 10 10")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.gui = gui  # Reference to main GUI for callbacks

        # Title
        title_label = ttk.Label(self, text="Gomoku AI", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="ew")

        # Current player indicator
        player_frame = ttk.Frame(self, padding="5 0")
        player_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        player_label = ttk.Label(player_frame, text="Current Player:", font=("Arial", 11))
        player_label.pack(side=tk.LEFT, padx=(0, 5))
        self.player_indicator = Canvas(player_frame, width=20, height=20, highlightthickness=0)
        self.player_indicator.pack(side=tk.LEFT)
        self.player_indicator.create_oval(2, 2, 18, 18, fill=BLACK, outline=BLACK)

        # Turn indicator
        self.turn_var = StringVar(value="Player's Turn")
        self.turn_label = ttk.Label(self, textvariable=self.turn_var, font=("Arial", 12, "bold"), anchor="center")
        self.turn_label.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        # Win probability section
        prob_frame = ttk.LabelFrame(self, text="AI Evaluation", padding="10 5")
        prob_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 5))
        prob_frame.columnconfigure(1, weight=1)

        # Black probability
        black_label = ttk.Label(prob_frame, text="Black Win%:", font=("Arial", 10))
        black_label.grid(row=0, column=0, sticky="w", pady=2)
        self.black_prob_var = DoubleVar(value=50)
        self.black_prob_bar = ttk.Progressbar(prob_frame, variable=self.black_prob_var, length=100, mode="determinate", maximum=100)
        self.black_prob_bar.grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.black_prob_value = ttk.Label(prob_frame, text="50.0%", font=("Arial", 10), width=6)
        self.black_prob_value.grid(row=0, column=2, sticky="e", pady=2)

        # White probability
        white_label = ttk.Label(prob_frame, text="White Win%:", font=("Arial", 10))
        white_label.grid(row=1, column=0, sticky="w", pady=2)
        self.white_prob_var = DoubleVar(value=50)
        self.white_prob_bar = ttk.Progressbar(prob_frame, variable=self.white_prob_var, length=100, mode="determinate", maximum=100)
        self.white_prob_bar.grid(row=1, column=1, sticky="ew", pady=2, padx=5)
        self.white_prob_value = ttk.Label(prob_frame, text="50.0%", font=("Arial", 10), width=6)
        self.white_prob_value.grid(row=1, column=2, sticky="e", pady=2)

        # AI settings section
        settings_frame = ttk.LabelFrame(self, text="AI Settings", padding="10 5")
        settings_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        settings_frame.columnconfigure(1, weight=1)

        # Model selection
        model_label = ttk.Label(settings_frame, text="Model:", font=("Arial", 10))
        model_label.grid(row=0, column=0, sticky="w", pady=3)
        self.model_var = StringVar(value=os.path.basename(MODEL_PATH))
        model_entry_frame = ttk.Frame(settings_frame)
        model_entry_frame.grid(row=0, column=1, sticky="ew", pady=3)
        model_entry = ttk.Entry(model_entry_frame, textvariable=self.model_var, width=18)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        model_button = ttk.Button(model_entry_frame, text="...", width=3, command=self.browse_model)
        model_button.pack(side=tk.LEFT, padx=(2, 0))

        # Temperature slider
        temp_label = ttk.Label(settings_frame, text="Temp:", font=("Arial", 10))
        temp_label.grid(row=1, column=0, sticky="w", pady=3)
        self.temp_var = DoubleVar(value=1.0)
        temp_slider = ttk.Scale(settings_frame, variable=self.temp_var, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        temp_slider.grid(row=1, column=1, sticky="ew", pady=3)

        # Heatmap toggle
        self.show_heatmap_var = BooleanVar(value=True)
        heatmap_check = ttk.Checkbutton(settings_frame, text="Show Heatmap", variable=self.show_heatmap_var)
        heatmap_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        
        # Value evaluation toggle
        self.use_value_eval_var = BooleanVar(value=False)
        value_eval_check = ttk.Checkbutton(
            settings_frame, 
            text="Use Value Evaluation", 
            variable=self.use_value_eval_var,
            command=self.toggle_value_eval
        )
        value_eval_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=3)
        
        # Add weighted approach toggle
        self.use_weighted_var = BooleanVar(value=True)
        weighted_check = ttk.Checkbutton(
            settings_frame, 
            text="Use Weighted Policy-Value", 
            variable=self.use_weighted_var,
            command=self.toggle_weighted
        )
        weighted_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        
        # Alpha slider (policy vs value weight)
        alpha_label = ttk.Label(settings_frame, text="Policy Weight:", font=("Arial", 10))
        alpha_label.grid(row=5, column=0, sticky="w", pady=3)
        
        alpha_frame = ttk.Frame(settings_frame)
        alpha_frame.grid(row=5, column=1, sticky="ew", pady=3)
        
        self.alpha_var = DoubleVar(value=0.7)
        alpha_slider = ttk.Scale(
            alpha_frame, 
            variable=self.alpha_var, 
            from_=0.1, 
            to=0.9, 
            orient=tk.HORIZONTAL,
            command=self.update_alpha
        )
        alpha_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.alpha_label = ttk.Label(alpha_frame, text="0.7", width=3)
        self.alpha_label.pack(side=tk.RIGHT, padx=5)
        
        # Top-K moves to evaluate slider
        topk_label = ttk.Label(settings_frame, text="Top-K Moves:", font=("Arial", 10))
        topk_label.grid(row=6, column=0, sticky="w", pady=3)
        
        topk_frame = ttk.Frame(settings_frame)
        topk_frame.grid(row=6, column=1, sticky="ew", pady=3)
        
        self.topk_var = IntVar(value=8)
        topk_slider = ttk.Scale(
            topk_frame, 
            variable=self.topk_var, 
            from_=3, 
            to=10, 
            orient=tk.HORIZONTAL,
            command=self.update_topk
        )
        topk_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.topk_label = ttk.Label(topk_frame, text="8", width=2)
        self.topk_label.pack(side=tk.RIGHT, padx=5)

        # Game controls section
        controls_frame = ttk.LabelFrame(self, text="Game Controls", padding="10 5")
        controls_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(2, weight=1)

        self.new_game_button = ttk.Button(controls_frame, text="New Game")
        self.new_game_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.undo_button = ttk.Button(controls_frame, text="Undo", state=tk.DISABLED)
        self.undo_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.switch_sides_button = ttk.Button(controls_frame, text="Switch Sides")
        self.switch_sides_button.grid(row=0, column=2, padx=2, pady=2, sticky="ew")

        # Accuracy Stats section
        self.stats_frame = ttk.LabelFrame(self, text="AI Prediction Stats", padding="10 5")
        self.stats_frame.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=10)
        self.stats_details = ttk.Label(self.stats_frame, text="No moves yet", font=("Arial", 9), justify=tk.LEFT)
        self.stats_details.pack(anchor=tk.NW, pady=(0, 5))
        self.grid_rowconfigure(8, weight=1) # Allow stats frame to expand vertically

        # Status bar at the bottom
        self.status_var = StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 2")
        self.status_bar.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    
    def browse_model(self):
        """Open file dialog to select a model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.model_var.set(os.path.basename(filename))
            return filename
        return None
    
    def update_player_indicator(self, is_black):
        """Update the player indicator to show current player"""
        self.player_indicator.delete("all")
        if is_black:
            self.player_indicator.create_oval(2, 2, 18, 18, fill=BLACK, outline=BLACK)
        else:
            self.player_indicator.create_oval(2, 2, 18, 18, fill=WHITE, outline=BLACK)
    
    def update_turn_indicator(self, is_player_turn):
        """Update the turn indicator"""
        if is_player_turn:
            self.turn_var.set("Player's Turn")
        else:
            self.turn_var.set("AI's Turn")
    
    def update_win_probability(self, win_probability):
        """Update the win probability bars"""
        # REMOVED: Update value score display
        # self.value_var.set(f"{win_probability:.3f}")

        # Convert value (-1 to 1) to win probability (0 to 100) for current player
        black_prob = (win_probability + 1) / 2 * 100
        white_prob = 100 - black_prob
        
        self.black_prob_var.set(black_prob)
        self.white_prob_var.set(white_prob)
        self.black_prob_value.config(text=f"{black_prob:.1f}%")
        self.white_prob_value.config(text=f"{white_prob:.1f}%")
    
    def update_accuracy_stats(self, total_moves, correct, top3, top5):
        """Update the accuracy statistics display"""
        if total_moves > 0:
            top1_pct = (correct / total_moves * 100)
            top3_pct = (top3 / total_moves * 100)
            top5_pct = (top5 / total_moves * 100)
            
            stats_text = f"Top-1: {top1_pct:.1f}% ({correct}/{total_moves})\n"
            stats_text += f"Top-3: {top3_pct:.1f}% ({top3}/{total_moves})\n"
            stats_text += f"Top-5: {top5_pct:.1f}% ({top5}/{total_moves})"
            
            self.stats_details.config(text=stats_text)
    
    def toggle_value_eval(self):
        """Toggle value evaluation on/off"""
        use_value = self.use_value_eval_var.get()
        if use_value:
            # Turn off weighted approach when using pure value
            self.use_weighted_var.set(False)
        
        self.gui.use_value_eval = use_value
        self.set_status(f"Value evaluation {'enabled' if use_value else 'disabled'}")
    
    def toggle_weighted(self):
        """Toggle weighted policy-value approach on/off"""
        use_weighted = self.use_weighted_var.get()
        if use_weighted:
            # Turn off pure value approach when using weighted
            self.use_value_eval_var.set(False)
            self.gui.use_value_eval = False
        
        self.set_status(f"Weighted policy-value {'enabled' if use_weighted else 'disabled'}")
    
    def update_alpha(self, value):
        """Update the policy weight for weighted approach"""
        value = float(value)
        self.alpha_label.config(text=f"{value:.1f}")
        self.set_status(f"Policy weight set to {value:.1f}, value weight to {1-value:.1f}")
    
    def update_topk(self, value):
        """Update the number of top moves to evaluate with value head"""
        self.value_eval_top_k = int(float(value))
        # Update the display label
        self.topk_label.config(text=str(int(float(value))))
    
    def set_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)

class GomokuGUI:
    def __init__(self, root):
        # Main window setup
        self.root = root
        self.root.title("Gomoku AI")
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        
        # Initialize UI variables and state
        self.board_size = 15
        self.cell_size = CELL_SIZE
        self.board_array = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        self.board_array[:, :, 2] = 1.0  # All positions start empty
        self.current_player = True  # True for Black, False for White
        self.is_player_turn = True  # Player goes first
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.last_move = None
        self.probabilities = None
        self.win_probability = None
        self.model = None
        self.device = 'cpu'
        self.temperature = 0.8  # Default temperature
        self.auto_play = False
        self.ai_vs_ai_mode = False
        self.ai_thinking = False
        self.use_value_eval = False  # Default to not use value-based evaluation
        self.value_eval_top_k = 8   # Number of top moves to evaluate
        
        # Create main frames
        self.main_frame = Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create board frame on the left
        self.board_frame = Frame(self.main_frame)
        self.board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create info panel on the right
        self.info_panel = InfoPanel(self.main_frame, self)
        self.info_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Create board canvas
        self.board = GomokuBoard(self.board_frame)
        self.board.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Connect board click event
        self.board.bind("<Button-1>", self.on_board_click)
        
        # Connect buttons to actions
        self.info_panel.new_game_button.config(command=self.new_game)
        self.info_panel.undo_button.config(command=self.undo_move)
        self.info_panel.switch_sides_button.config(command=self.switch_sides)
        
        # Load the model
        self.load_model()
        
        # Start a new game
        self.new_game()
        
        # Keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self.undo_move())
        self.root.bind("<Control-n>", lambda e: self.new_game())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        
        # Update status
        self.info_panel.set_status("Ready to play! Click to place a stone.")
    
    def load_model(self, model_path=None):
        """Load the AI model"""
        if model_path is None:
            # Use the path from the InfoPanel
            model_path_from_gui = self.info_panel.model_var.get()
            # Construct full path if it's just a basename
            if not os.path.dirname(model_path_from_gui):
                model_path_from_gui = os.path.join(os.getcwd(), model_path_from_gui)
            model_path = model_path_from_gui if os.path.exists(model_path_from_gui) else MODEL_PATH

        try:
            # Check if the file exists before attempting to load
            if not os.path.exists(model_path):
                 self.info_panel.set_status(f"Error: Model file not found at {model_path}")
                 messagebox.showerror("Error", f"Model file not found: {model_path}")
                 return False
                 
            self.info_panel.set_status(f"Loading: {os.path.basename(model_path)}...")
            self.root.update_idletasks() # Force UI update
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = load_model(model_path, device=self.device)
            # Update the model var in case a default was loaded
            self.info_panel.model_var.set(os.path.basename(model_path))
            self.info_panel.set_status(f"Model Ready: {os.path.basename(model_path)} ({self.device.upper()})")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model '{os.path.basename(model_path)}':\n{str(e)}")
            self.info_panel.set_status("Error loading model")
            return False
    
    def new_game(self):
        """Start a new game"""
        # Clear board visually
        self.board.clear_board()
        
        # Reset game state
        self.board_array = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        self.board_array[:, :, 2] = 1.0  # All positions start as empty
        self.current_player = True  # Black goes first
        self.is_player_turn = self.current_player  # Player goes first if they play black
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.move_history = []
        self.probabilities = np.zeros((BOARD_SIZE, BOARD_SIZE))
        
        # Reset AI accuracy tracking
        self.ai_total_moves = 0
        self.ai_correct_predictions = 0
        self.ai_top3_correct = 0
        self.ai_top5_correct = 0
        self.info_panel.update_accuracy_stats(0, 0, 0, 0)
        
        # Update UI
        self.info_panel.update_player_indicator(True)  # Black goes first
        self.info_panel.update_turn_indicator(self.is_player_turn)
        self.info_panel.update_win_probability(0.0) # Reset value to 0
        self.info_panel.undo_button.config(state=tk.DISABLED)
        
        # If AI goes first, make the first move
        if not self.is_player_turn:
            self.info_panel.set_status("AI starts. Thinking...")
            self.root.after(100, self.make_ai_move) # Slight delay
        else:
            player_color = "Black" if self.current_player else "White"
            self.info_panel.set_status(f"New Game. Player ({player_color}) starts.")
        
        self.info_panel.set_status("New game started")
    
    def switch_sides(self):
        """Switch between playing as black or white"""
        self.current_player = not self.current_player
        # Start a new game with switched sides
        self.new_game()
        
        side = "Black" if self.current_player else "White"
        self.info_panel.set_status(f"Switched sides. Player is now {side}")
    
    def undo_move(self):
        """Undo the last move(s) - usually player + AI move"""
        # Determine how many moves to undo (2 if player and AI moved, 1 if only player moved)
        moves_to_undo = 0
        if self.move_history:
            last_player_made_move = self.current_player if len(self.move_history) % 2 != 0 else not self.current_player
            if self.is_player_turn != last_player_made_move: # If turn switched, likely AI also moved
                moves_to_undo = 2
            else: # Only player moved
                moves_to_undo = 1
        
        # Ensure we don't undo more moves than available
        moves_to_undo = min(moves_to_undo, len(self.move_history))

        if moves_to_undo == 0 or self.game_over:
             self.info_panel.set_status("Cannot undo move.")
             return

        # Undo the moves
        undone_moves = []
        for _ in range(moves_to_undo):
            if self.move_history:
                row, col = self.move_history.pop()
                undone_moves.append((row, col))
                # Clear the position in the board array
                self.board_array[row, col, 0] = 0.0  # Clear black
                self.board_array[row, col, 1] = 0.0  # Clear white
                self.board_array[row, col, 2] = 1.0  # Mark as empty
                self.current_player = not self.current_player # Toggle player back for each undone move
        
        # Update last move indicator
        self.last_move = self.move_history[-1] if self.move_history else None
        
        # Redraw the entire board with current state
        self.redraw_board()
        
        # Reset turn to player
        self.is_player_turn = True 
        
        # Update UI
        self.info_panel.update_player_indicator(self.current_player)
        self.info_panel.update_turn_indicator(self.is_player_turn)
        self.info_panel.update_win_probability(0.0) # Reset value after undo
        self.info_panel.show_heatmap_var.set(False) # Hide heatmap after undo

        # Disable undo button if no more moves (or only 1 left)
        self.info_panel.undo_button.config(state=tk.DISABLED if len(self.move_history) < 1 else tk.NORMAL)

        self.info_panel.set_status(f"Undid {moves_to_undo} move(s).")
    
    def on_board_click(self, event):
        """Handle click on the board"""
        if self.game_over:
            self.info_panel.set_status("Game is over. Start a new game.")
            return
        if not self.is_player_turn:
            self.info_panel.set_status("Please wait for AI turn.")
            return

        # Convert click coordinates to board position
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        # Check if the position is valid and empty
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        if self.board_array[row, col, 2] != 1.0:  # Position not empty
            self.info_panel.set_status("Invalid move: Position occupied.")
            return
        
        player_color = "Black" if self.current_player else "White"
        self.info_panel.set_status(f"Player ({player_color}) placed at ({row+1}, {col+1}).")

        # Make the move
        self.make_move(row, col)

        # If game is not over, let AI make its move
        if not self.game_over:
             self.info_panel.set_status("AI thinking...")
             self.root.after(100, self.make_ai_move)  # Reduced delay
    
    def make_move(self, row, col):
        """Make a move at the specified position (internal logic)"""
        # Place the stone
        channel = 0 if self.current_player else 1  # Black = 0, White = 1
        self.board_array[row, col, channel] = 1.0
        self.board_array[row, col, 2] = 0.0  # No longer empty

        # Update last move and history
        self.last_move = (row, col)
        self.move_history.append((row, col))

        # Enable undo button (allow undoing even the first move)
        self.info_panel.undo_button.config(state=tk.NORMAL)

        # Redraw the board
        self.redraw_board()

        # Check if the game is over
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            winner_text = "Black" if self.current_player else "White"
            self.info_panel.set_status(f"Game Over! {winner_text} wins!")
            self.info_panel.update_turn_indicator(False) # Indicate game ended
            messagebox.showinfo("Game Over", f"{winner_text} wins!")
            return

        # Check if the board is full (draw)
        if np.sum(self.board_array[:, :, 2]) == 0:
            self.game_over = True
            self.info_panel.set_status("Game Over - Draw!")
            self.info_panel.update_turn_indicator(False) # Indicate game ended
            messagebox.showinfo("Game Over", "Draw!")
            return

        # Switch player
        self.current_player = not self.current_player
        self.is_player_turn = not self.is_player_turn

        # Update UI
        self.info_panel.update_player_indicator(self.current_player)
        self.info_panel.update_turn_indicator(self.is_player_turn)

    def make_ai_move(self):
        """Let the AI make a move"""
        if self.game_over or self.ai_thinking:
            return
            
        # Set thinking flag to prevent multiple calls
        self.ai_thinking = True
        
        # Get device and temperature from settings
        device = self.device
        temperature = self.info_panel.temp_var.get()
        
        # Update status to indicate AI is thinking
        ai_color = "Black" if self.current_player else "White"
        self.info_panel.set_status(f"AI ({ai_color}) is thinking...")
        self.root.update()  # Ensure status update is visible

        try:
            # Calculate current move count
            move_count = len(self.move_history)

            # Get value evaluation and weighted settings
            use_value_eval = self.use_value_eval
            use_weighted = self.info_panel.use_weighted_var.get()
            top_k = self.value_eval_top_k
            alpha = self.info_panel.alpha_var.get()
            
            # Choose prediction method based on settings
            if use_weighted:
                # Make prediction with weighted approach
                top_move, probs, value, evaluated_moves = predict_move_with_weighted_details(
                    self.model, 
                    self.board_array,
                    temperature=temperature,
                    alpha=alpha,
                    move_count=move_count,
                    device=device,
                    top_k=top_k
                )
                method_desc = f"weighted (α={alpha:.1f})"
            else:
                # Make prediction with original approach
                top_move, probs, value, evaluated_moves = predict_move_with_details(
                    self.model, 
                    self.board_array,
                    temperature=temperature,
                    move_count=move_count,
                    device=device,
                    top_k=top_k,
                    use_value_eval=use_value_eval
                )
                method_desc = "value" if use_value_eval else "policy"
            
            # Update win probability
            self.win_probability = value
            self.info_panel.update_win_probability(value)

            # Update UI with probabilities and evaluated moves if requested
            if self.info_panel.show_heatmap_var.get():
                self.probabilities = probs
                self.evaluated_moves = evaluated_moves
                # Redraw board happens in make_move, heatmap drawn there

            # Convert the move to 0-indexed
            row, col = top_move[0] - 1, top_move[1] - 1
            
            ai_color = "Black" if self.current_player else "White"
            self.info_panel.set_status(f"AI ({ai_color}) placed at ({row+1}, {col+1}) using {method_desc} method")

            # Make the move
            self.make_move(row, col)
            
            # If game didn't end, set status for player's turn
            if not self.game_over:
                player_color = "Black" if self.current_player else "White"
                self.info_panel.set_status(f"Player's turn ({player_color}).")

        except Exception as e:
            error_msg = f"AI move error: {str(e)}"
            messagebox.showerror("AI Error", error_msg)
            self.info_panel.set_status(f"Error: {error_msg}")
            # Consider giving turn back to player or attempting random move
            self.is_player_turn = True
            self.info_panel.update_turn_indicator(self.is_player_turn)
        
        finally:
            # Reset thinking flag
            self.ai_thinking = False
            
            # If in AI vs AI mode and game not over, schedule next move
            if self.ai_vs_ai_mode and not self.game_over:
                self.root.after(1000, self.make_ai_move)
    
    def redraw_board(self):
        """Redraw the entire board with current state"""
        self.board.clear_board()
        
        # Share board array with the board widget
        self.board.board_array = self.board_array
        
        # Draw probability heatmap if enabled (draw FIRST, so stones appear on top)
        if self.info_panel.show_heatmap_var.get() and not self.game_over:
            self.board.draw_probability_heatmap(self.probabilities)
        
        # Draw all stones
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board_array[row, col, 0] == 1.0:  # Black stone
                    is_last = self.last_move == (row, col)
                    self.board.draw_stone(row, col, True, is_last)
                elif self.board_array[row, col, 1] == 1.0:  # White stone
                    is_last = self.last_move == (row, col)
                    self.board.draw_stone(row, col, False, is_last)
    
    def check_win(self, row, col):
        """Check if the last move at (row, col) resulted in a win"""
        # Determine the player who made the move
        is_black = self.board_array[row, col, 0] == 1.0
    
        # Directions: horizontal, vertical, diagonal (\), diagonal (/)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # Start with 1 for the current position
            
            # Check in both directions along the current line
            for direction in [1, -1]:
                for i in range(1, 5):  # Look for 5 in a row
                    nx, ny = row + direction * i * dx, col + direction * i * dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if is_black and self.board_array[nx, ny, 0] == 1.0:  # Black stone
                            count += 1
                        elif not is_black and self.board_array[nx, ny, 1] == 1.0:  # White stone
                            count += 1
                        else:
                            break
                    else:
                        break
                    
            if count >= 5:
                # Highlight the winning line
                self.highlight_winning_line(row, col)
                return True
            
        return False

    def highlight_winning_line(self, row, col):
        """Highlight the winning line of 5 stones with a single red line"""
        # Determine the player who made the move
        is_black = self.board_array[row, col, 0] == 1.0
        stone_channel = 0 if is_black else 1
        
        # Directions: horizontal, vertical, diagonal (\), diagonal (/)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            line_positions = [(row, col)]
            line_length = 1
            
            # Check in positive direction
            for i in range(1, 5):
                nx, ny = row + i * dx, col + i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board_array[nx, ny, stone_channel] == 1.0:
                    line_positions.append((nx, ny))
                    line_length += 1
                else:
                    break
            
            # Check in negative direction
            for i in range(1, 5):
                nx, ny = row - i * dx, col - i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board_array[nx, ny, stone_channel] == 1.0:
                    line_positions.insert(0, (nx, ny)) # Insert at beginning for correct order
                    line_length += 1
                else:
                    break
            
            # If we found 5 or more in a row, draw the line through centers
            if line_length >= 5:
                # Sort positions to ensure line is drawn correctly (esp. for diagonals)
                line_positions.sort()
                
                # Get start and end coordinates for the line
                start_row, start_col = line_positions[0]
                end_row, end_col = line_positions[line_length - 1] # Use actual length
                
                start_x = start_col * CELL_SIZE + CELL_SIZE // 2
                start_y = start_row * CELL_SIZE + CELL_SIZE // 2
                end_x = end_col * CELL_SIZE + CELL_SIZE // 2
                end_y = end_row * CELL_SIZE + CELL_SIZE // 2
                
                # Draw a single red line through the center of the winning stones
                self.board.create_line(start_x, start_y, end_x, end_y, \
                                     width=4, fill=RED, tags="win_highlight", capstyle=tk.ROUND)
                return True # Only highlight the first winning line found
        
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gomoku GUI")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                      help="Path to the model to use")
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    
    # Create the Tkinter window
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()