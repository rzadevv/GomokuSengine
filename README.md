# GomokuSengine

A deep learning-based artificial intelligence system for playing Gomoku (Five in a Row) with sophisticated neural network architecture, training pipeline, and testing framework. The model utilizes a state-of-the-art convolutional neural network featuring residual blocks with dilated convolutions, multi-scale feature extraction, spatial and channel attention mechanisms, and non-local blocks to capture complex board patterns across different spatial scales. This project combines modern deep learning techniques with traditional game strategies to create a powerful Gomoku AI engine, leveraging a dual-headed network architecture that simultaneously predicts move probabilities through a policy head and evaluates board positions through a value head, all trained on expert human games with data augmentation and optimized using mixed-precision training.

## 🔧 Installation

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
git clone https://github.com/rzadevv/GomokuSengine.git
cd GomokuSengine
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

## 🕹️ Usage

### Play Against the AI

```bash
python main.py gui --model_path model0.5xdropout.pth
```
Optional parameters:
- `--temperature`: Set the exploration temperature (default: 1.0)
- `--use_value_eval`: Enable value-based move evaluation

### 🧠 Train a New Model

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

### 🧬 Neural Network

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

### 🏗 Training Pipeline

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

### 🧮 Inference Engine
The inference engine employs several strategies to balance strong play with reasonable response time:
- **Base Move Prediction**: Initial policy head output gives move probabilities
- **Value-guided Selection**:
  - For top-k moves from policy, simulate each move and evaluate resulting position
  - Choose move with best expected outcome based on value predictions
  - Configurable with `--top_k` parameter (default: 8)

- **Temperature-based Exploration**: 
  - Lower temperature → more deterministic (best moves)
  - Higher temperature → more exploration (diverse moves)
  - Automatically adjusted based on game phase

- **Move Weighting Technique**:
  - Combines policy probabilities and value estimates with configurable alpha parameter
  - Formula: `final_score = alpha * policy_score + (1 - alpha) * value_score`
  - Adjustable with `--alpha` parameter (default: 0.7)
