# GomokuSengine: Advanced Neural Network Gomoku AI

A deep learning-based artificial intelligence system for playing Gomoku (Five in a Row) with sophisticated neural network architecture, training pipeline, and testing framework. This project combines modern deep learning techniques with traditional game strategies to create a powerful Gomoku AI engine.


## Features

- **Deep Learning Architecture**: CNN-based neural network with residual blocks, multi-scale feature extraction, attention mechanisms, and non-local blocks
- **Dual-headed Output**: Policy head for move prediction and value head for position evaluation
- **Advanced Training Pipeline**: Distributed multi-GPU training with mixed precision and early stopping
- **Rich Visualization Tools**: Real-time visualization of model predictions, accuracy metrics, and performance
- **Comprehensive Testing Framework**: Engine comparison, pattern recognition testing, and performance benchmarking
- **Interactive GUI**: Play against the AI with adjustable parameters and visualization
- **Performance Optimization**: Tuned for efficient inference on both CPU and GPU

## Technical Overview

GomokuSengine implements a two-headed convolutional neural network that combines pattern recognition capabilities with strategic evaluation. The system uses:

- **3-Channel Board Representation**: Black stones (channel 0), white stones (channel 1), and empty positions (channel 2)
- **Temperature-based Exploration**: Adjustable temperature parameter to balance exploitation and exploration
- **Value-guided Search**: Combines policy predictions with position evaluation for stronger decision making
- **Pattern Recognition**: Specialized testing framework validates the model's ability to recognize critical Gomoku patterns like open threes, open fours, and forcing sequences
- **Performance Benchmarking**: GFLOPS calculation and memory profiling to optimize real-time performance

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)
- NumPy, Matplotlib, tqdm
- Pygame (for visualization)
- Additional dependencies in `requirements.txt`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sgoldnn.git
cd sgoldnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract training data from RAR archive:
```bash
# Extract the psq_games.rar file containing gomoku game records
# For Windows, use WinRAR or 7-Zip
# For Linux/Mac:
unrar x psq_games.rar
```

4. Generate the training dataset (required before training):
```bash
python main.py data --psq_dir psq_games --output gomoku_dataset.npz
```

## Usage

### Play Against the AI

```bash
python main.py gui --model_path model0.5xdropout.pth
```
Optional parameters:
- `--temperature`: Set the exploration temperature (default: 1.0)
- `--use_value_eval`: Enable value-based move evaluation

### Train a New Model

```bash
# First ensure you've generated the dataset (see Setup step 4)
python main.py train --npz gomoku_dataset.npz --epochs 20 --batch_size 128 --lr 0.001
```

Advanced training parameters:
- `--smoothing`: Label smoothing factor (default: 0.1)
- `--policy_weight`: Weight for policy loss component (default: 1.0)
- `--value_weight`: Weight for value loss component (default: 1.3)
- `--distributed`: Enable distributed multi-GPU training
- `--stable_lr`: Use constant learning rate instead of OneCycleLR


## Architecture

### Neural Network

The GomokuNet architecture consists of:

- **Input Layer**: Processes 3-channel board representation (15×15×3)
- **Initial Convolution**: Raises input from 3 channels (black, white, empty) to 64 feature channels
- **Residual Blocks**: 3 blocks with increasing dilation rates (1, 2, 4) for broader receptive fields
- **Multi-Scale Feature Extraction**: Processes board patterns at different scales by using parallel convolutions with different dilation rates (1, 2, 4)
- **Attention Mechanisms**: 
  - Channel attention: Re-weights feature channels based on importance
  - Spatial attention: Focuses on important board regions
- **Non-local Block**: Captures long-range dependencies by modeling relationships between all positions
- **Dual Output Heads**:
  - Policy head: Outputs probability distribution over all 225 board positions (15×15)
  - Value head: Estimates win probability for current player in range [-1, 1]

### Training Pipeline

- **Data Pipeline**: 
  - Processes .psq game records with data augmentation (rotations, flips)
  - Analyzes game phases (opening, midgame, endgame)
  - Creates balanced training samples by game phase

- **Custom Dataset**: Implements `GomokuDataset` with stratified sampling by game phase
  - Each sample includes:
    - Board state (3 channels: black, white, empty)
    - Target move (one-hot encoded)
    - Game phase (opening, midgame, endgame)
    - Value target (-1 to 1)

- **Loss Function**: Combined policy (cross-entropy) and value (MSE) loss

- **Optimization**: 
  - AdamW optimizer for better weight decay handling
  - OneCycleLR scheduler for faster convergence
  - Optional stable learning rate for fine-tuning

- **Early Stopping**: Prevents overfitting with validation accuracy monitoring
  - Patience parameter (default: 5 epochs)
  - Saves best model based on validation accuracy

- **Distributed Training**: 
  - Optional multi-GPU distributed training using PyTorch DDP
  - Implements efficient data sharding across GPUs

### Inference Engine
The inference engine employs several strategies to balance strong play with reasonable response time:
- **Base Move Prediction**: Initial policy head output gives move probabilities
- **Value-guided Selection**:
  - For top-k moves from policy, simulate each move and evaluate resulting position
  - Choose move with best expected outcome based on value predictions
  - Configurable with `--top_k` parameter (default: 5)

- **Temperature-based Exploration**: 
  - Lower temperature → more deterministic (best moves)
  - Higher temperature → more exploration (diverse moves)
  - Automatically adjusted based on game phase

- **Move Weighting Technique**:
  - Combines policy probabilities and value estimates with configurable alpha parameter
  - Formula: `final_score = alpha * policy_score + (1 - alpha) * value_score`
  - Adjustable with `--alpha` parameter (default: 0.7)

## Directory Structure

```
sgoldnn/
├── main.py                  # Main entry point
├── model.py                 # Neural network architecture
├── train.py                 # Training implementation
├── train_gui.py             # GUI for training configuration
├── datapipeline.py          # Game record processing
├── inference.py             # Model inference code
├── visualization.py         # Plotting and visualization tools
├── gui.py                   # Game GUI for playing against AI
├── logger.py                # Logging utilities
├── config.yaml              # Configuration settings
├── vals8engine/             # Engine testing and evaluation
│   ├── model_battle.py      # Compare different models
│   ├── battle_mcts.py       # Test against MCTS algorithm
│   ├── pattern_test_mcts.py # Test pattern recognition vs MCTS
│   ├── pattern_test_opponent_mmai.py # Pattern recognition vs opponent's engine
│   ├── pattern_test_developed engine..py # Pattern recognition with developed engine
│   ├── engine_battle_mm_ai.py # Battle against opponent's minimax AI
│   ├── ablation_battle.py   # Ablation study for architecture components
│   ├── calculate_latency_developedengine.py # Latency measurements
│   ├── Gflopscal.py         # Performance benchmarking (GFLOPS)
│   └── memory_profiler.py   # Memory usage monitoring
├── gomokumcts/              # MCTS implementation
├── Opponent_gomoku_engine/  # Alternative engine for comparison
├── psq_games.rar            # Compressed game records (extract before training)
└── docs/                    # Documentation
```
