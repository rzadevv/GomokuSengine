import os
import time
import numpy as np
from collections import Counter
import tqdm
import datetime
import sys
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler, random_split
import torch.multiprocessing as mp
import torch.nn.functional as F

from model import GomokuNet
from logger import get_training_logger
from visualization import TrainingVisualizer

logger = get_training_logger()

if torch.cuda.is_available() and os.name == 'nt':
    torch.multiprocessing.set_start_method('spawn', force=True)

# -------------------------------
# 1. Dataset & Stratified Sampler
# -------------------------------

class GomokuDataset(Dataset):
    """
    Loads pre-processed data from an .npz file.
    Each sample returns:
      - board: Tensor (3, 15, 15)
      - target: One-hot tensor (15, 15)
      - phase: string ('opening', 'midgame', 'endgame')
      - value: Value target (-1 to 1) indicating win probability
    """
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.boards = data['boards']      # shape: (N, 15, 15, 3)
        self.targets = data['targets']      # shape: (N, 15, 15)
        self.phases = data['phases']        # shape: (N,)
        self.values = data['values']        # shape: (N,)
        
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        board = torch.tensor(self.boards[idx], dtype=torch.float32)
        # Rearrange to (channels, height, width)
        board = board.permute(2, 0, 1)
        
        # Convert 2D one-hot target to 1D flattened target for 225-class output
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        # Ensure it's properly one-hot encoded when flattened
        target = target.view(-1)  # Flatten to 225 dimensions
        
        # Get value target
        value = torch.tensor(self.values[idx], dtype=torch.float32).view(1)
        
        phase = self.phases[idx]
        return board, target, value, phase

class StratifiedSampler(torch.utils.data.Sampler):
    """
    Stratified sampling by game phase to ensure balanced representation
    in both training and validation sets.
    """
    def __init__(self, dataset, validation_split=0.2, is_validation=False, shuffle=True):
        self.dataset = dataset
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.shuffle = shuffle
        
        # Group indices by game phase
        self.phase_indices = {
            'opening': [],
            'midgame': [],
            'endgame': []
        }
        
        for idx in range(len(dataset)):
            _, _, _, phase = dataset[idx]
            if phase in self.phase_indices:
                self.phase_indices[phase].append(idx)
        
        self.indices = []
        for phase, indices in self.phase_indices.items():
            phase_size = len(indices)
            if phase_size == 0:
                continue
                
            generator = torch.Generator().manual_seed(42)
            
            val_size = int(phase_size * validation_split)
            train_size = phase_size - val_size
            
            phase_train, phase_val = random_split(indices, [train_size, val_size], generator=generator)
            
            if is_validation:
                self.indices.extend(phase_val)
            else:
                self.indices.extend(phase_train)
    
    def __iter__(self):
        if self.shuffle:
            random.seed(42)  
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

def compute_sample_weights(phases):
    """
    Compute sample weights inversely proportional to the frequency of each phase,
    ensuring balanced sampling across 'opening', 'midgame', and 'endgame'.
    """
    phase_counts = Counter(phases)
    weights = [1.0 / phase_counts[phase] for phase in phases]
    return weights

def create_train_val_datasets(full_dataset, val_split, rank, world_size):
    """
    Split dataset into training and validation sets for distributed training.
    
    Args:
        full_dataset: The complete dataset
        val_split: Fraction to use for validation
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        train_dataset, val_dataset
    """
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    return train_dataset, val_dataset

# ---------------------
# Model Architecture is now imported from model.py
# ---------------------

# -------------------------
# Combined Loss Function
# -------------------------

class GomokuLoss(nn.Module):
    def __init__(self, policy_weight=1.0, value_weight=1.0, label_smoothing=0.1, num_classes=225):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
    def forward(self, policy_logits, value_pred, policy_target, value_target):
        """
        Calculate the combined policy and value loss.
        
        Args:
            policy_logits: Model policy output (B, num_classes)
            value_pred: Model value output (B, 1)
            policy_target: Target policy distribution (B, num_classes)
            value_target: Target value (B, 1)
            
        Returns:
            tuple: (total_loss, policy_loss, value_loss)
        """
        if self.label_smoothing > 0:
            smooth_target = policy_target * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            smooth_target = policy_target
        
        policy_loss = F.cross_entropy(policy_logits, smooth_target)
        value_loss = F.mse_loss(value_pred, value_target)
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss       
        return total_loss, policy_loss, value_loss

# --------------------------
# 4. Early Stopping
# --------------------------

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stop the training when validation accuracy is not improving for a given patience.
    """
    def __init__(self, patience=5, min_delta=0, verbose=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.mode = mode  # 'min' for loss, 'max' for accuracy
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.mode == 'min':
            improved = val_score < self.best_score - self.min_delta
        else:
            improved = val_score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = val_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} - Best: {self.best_score:.4f}, Current: {val_score:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# --------------------------
# 5. Multi-GPU Training Setup
# --------------------------

def setup_distributed(rank, world_size):
    """
    Setup distributed training environment with robust error handling.
    
    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        timeout = 30.0
        logger.info(f"Initializing process group (rank={rank}, world_size={world_size}, backend={backend})")
        
        dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=timeout))
        logger.info(f"Process group initialized successfully")
        
        tensor = torch.zeros(1).to(rank if torch.cuda.is_available() else "cpu")
        if rank == 0:
            for i in range(1, world_size):
                dist.send(tensor, dst=i)
                logger.info(f"Master sent verification to rank {i}")
        else:
            dist.recv(tensor, src=0)
            logger.info(f"Rank {rank} received verification from master")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            logger.info(f"Rank {rank} using GPU: {torch.cuda.get_device_name()}")
        else:
            logger.warning(f"Rank {rank} using CPU (CUDA not available)")
        
        dist.barrier()
        logger.info(f"Rank {rank} completed setup_distributed")
        return True
        
    except RuntimeError as e:
        error_msg = str(e)
        if "Address already in use" in error_msg:
            logger.error(f"Port 12355 already in use. Try terminating existing processes or use a different port.")
        elif "NCCL" in error_msg and "timeout" in error_msg:
            logger.error(f"NCCL timeout error. Check GPU connectivity and system resources.")
        elif "NCCL" in error_msg and "unhandled cuda error" in error_msg:
            logger.error(f"CUDA error in NCCL. Check GPU health and CUDA installation.")
        else:
            logger.error(f"Failed to initialize {backend} backend: {error_msg}")
        
        logger.warning("Falling back to single process training")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in distributed setup: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Falling back to single process training")
        return False

def cleanup_distributed():
    """
    Cleanup the distributed environment with error handling.
    """
    try:
        if dist.is_initialized():
            logger.info("Destroying process group")
            dist.destroy_process_group()
    except Exception as e:
        logger.error(f"Error during distributed cleanup: {str(e)}")

# --------------------------
# 6. Training Functionality
# --------------------------

def safe_visualize(visualizer, **kwargs):
    """Safely call visualization methods with error handling."""
    if not visualizer:
        return
        
    try:
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                processed_kwargs[key] = np.array(value)
            else:
                processed_kwargs[key] = value
        mapped_kwargs = {
            'value_preds': processed_kwargs.get('value_preds'),
            'value_targets': processed_kwargs.get('value_targets'),
            'top3_accuracy': processed_kwargs.get('top3_accuracy'),
            'top5_accuracy': processed_kwargs.get('top5_accuracy'),
            'phase_accuracy': processed_kwargs.get('phase_accuracy')
        }
        return visualizer.save_all_plots(**mapped_kwargs)
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print("Training visualizations saved to: figures/")

def train_model(
    npz_path,
    model_path="gomoku_model.pth",
    load_model=None,
    board_size=15,
    validation_split=0.2,
    batch_size=64,
    epochs=10,
    lr=0.001,
    patience=5,
    policy_weight=1.0,
    value_weight=1.3,
    label_smoothing=0.1,
    use_stable_lr=False,
    verbose=True,
    visualize=True,
    device="cpu",
    rank=0,
    world_size=1,
    distributed=False,
    random_seed=42,
):
    """
    Trains a Gomoku policy-value network.
    
    Args:
        npz_path (str): Path to the NPZ file containing move data.
        model_path (str): Path where the trained model will be saved.
        load_model (str, optional): Path to an existing model to continue training.
        board_size (int): Size of the Gomoku board.
        validation_split (float): Proportion of data to use for validation.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        policy_weight (float): Weight for the policy loss.
        value_weight (float): Weight for the value loss.
        label_smoothing (float): Label smoothing factor.
        use_stable_lr (bool): If True, use a stable learning rate without scheduling.
        verbose (bool): Whether to print verbose output.
        visualize (bool): Whether to create visualization plots.
        device (str): Device to use for training ('cpu' or 'cuda').
        rank (int): Process rank for distributed training.
        world_size (int): Total number of processes for distributed training.
        distributed (bool): Whether to use distributed training.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        model (GomokuNet): The trained model.
        train_losses (list): Training losses.
        val_losses (list): Validation losses.
        train_accuracies (list): Training accuracies.
        val_accuracies (list): Validation accuracies.
    """
    # Initialize TensorBoard if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        use_tensorboard = True
        writer = SummaryWriter('runs/gomoku_training')
        if rank == 0 or not distributed:
            print("TensorBoard initialized, logs in 'runs/gomoku_training'")
    except ImportError:
        use_tensorboard = False
        writer = None
        if rank == 0 or not distributed:
            print("TensorBoard not available, skipping TensorBoard logging")
    
    local_logger = get_training_logger() if rank == 0 or not distributed else None
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if local_logger:
        local_logger.info(f"Using device: {device}")
    
    if local_logger:
        local_logger.info(f"Loading data from {npz_path}")
    
    full_dataset = GomokuDataset(npz_path)
    
    if distributed:
        train_dataset, val_dataset = create_train_val_datasets(
            full_dataset, validation_split, rank, world_size)
    else:
        train_sampler = StratifiedSampler(full_dataset, validation_split=validation_split, is_validation=False, shuffle=True)
        val_sampler = StratifiedSampler(full_dataset, validation_split=validation_split, is_validation=True, shuffle=False)
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_sampler.indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_sampler.indices)
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,  
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )
    
    if local_logger:
        local_logger.info(f"Train dataset size: {len(train_dataset)}")
        local_logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    model = GomokuNet(board_size=board_size).to(device)
    
    visualizer = None
    if not distributed or rank == 0:
        try:
            from visualization import TrainingVisualizer
            visualizer = TrainingVisualizer(output_dir="figures")
        except ImportError:
            if local_logger:
                local_logger.warning("TrainingVisualizer not found, skipping visualization")
    
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank] if device=='cuda' else None,
            output_device=rank if device=='cuda' else None,
            find_unused_parameters=False
        )
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-4  
    )

    effective_batch_size = batch_size * world_size if distributed else batch_size
    
    if not use_stable_lr:
        total_steps = epochs * len(train_loader)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 0.8,  
            total_steps=total_steps,
            pct_start=0.45,  
            anneal_strategy='cos', 
            div_factor=20.0,  
            final_div_factor=1000.0  
        )
    else:
        scheduler = None
    criterion = GomokuLoss(policy_weight=policy_weight, value_weight=value_weight,
                         label_smoothing=label_smoothing)
    
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda' if device=='cuda' else 'cpu')
    else:
        scaler = torch.cuda.amp.GradScaler()
    
    early_stopping = EarlyStopping(patience=patience, mode='max', verbose=not distributed or rank == 0)
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    use_accuracy_metric = True  
    
    if not distributed or rank == 0:
        print(f"\nStarting training for {epochs} epochs:")
        print(f"{'-'*80}")
        print(f"Epoch |  Train Loss  | Train P-Loss | Train V-Loss |   Val Loss   |  Val P-Loss  |  Val V-Loss  |   Accuracy   |     LR      ")
        print(f"{'-'*80}")
    
    scaler = torch.cuda.amp.GradScaler()
    all_true_indices = []
    all_pred_indices = []
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    top3_accuracies = []
    top5_accuracies = []
    
    all_value_preds = []
    all_value_targets = []
    all_boards = []
    all_policy_outputs = []
    gradient_norms = []
    
    for epoch in range(epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        epoch_start_time = time.time()
        train_samples_count = 0
        
        pbar = None
        if not distributed or rank == 0:
            pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        correct_moves = 0
        total_moves = 0
        top3_correct = 0
        top5_correct = 0
        
        for boards, targets, values, _ in train_loader:
            boards = boards.to(device)  # shape: (B, 3, 15, 15)
            targets = targets.to(device)  # shape: (B, 225)
            values = values.to(device)   # shape: (B, 1)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu'):
                policy_logits, value_pred = model(boards)  # shapes: (B, 225), (B, 1)
                loss, p_loss, v_loss = criterion(policy_logits, value_pred, targets, values)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * boards.size(0)
            train_policy_loss += p_loss.item() * boards.size(0)
            train_value_loss += v_loss.item() * boards.size(0)
            train_samples_count += boards.size(0)
            
            if scheduler and not use_stable_lr:
                scheduler.step()
            
            pred_moves = torch.argmax(policy_logits, dim=1)
            true_moves = torch.argmax(targets, dim=1)
            correct_batch = (pred_moves == true_moves).sum().item()
            correct_moves += correct_batch
            total_moves += boards.size(0)
            
            if len(all_true_indices) < 100:  
                all_true_indices.extend(true_moves.cpu().numpy())
                all_pred_indices.extend(pred_moves.cpu().numpy())
            
            _, top_k_indices = torch.topk(policy_logits, k=5, dim=1)
            for i in range(len(true_moves)):
                if true_moves[i] in top_k_indices[i, :3]:
                    top3_correct += 1
                if true_moves[i] in top_k_indices[i, :5]:
                    top5_correct += 1
            
            if pbar:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "p_loss": f"{p_loss.item():.4f}", 
                    "v_loss": f"{v_loss.item():.4f}"
                })
            pbar.update(1)
        
        if pbar:
            pbar.close()
        
        if distributed:
            train_loss_tensor = torch.tensor([train_loss], device=device)
            train_policy_loss_tensor = torch.tensor([train_policy_loss], device=device)
            train_value_loss_tensor = torch.tensor([train_value_loss], device=device)
            train_samples_tensor = torch.tensor([train_samples_count], device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_policy_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_value_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_samples_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item()
            train_policy_loss = train_policy_loss_tensor.item()
            train_value_loss = train_value_loss_tensor.item()
            train_samples_count = train_samples_tensor.item()
        
        train_epoch_loss = train_loss / train_samples_count
        train_epoch_policy_loss = train_policy_loss / train_samples_count
        train_epoch_value_loss = train_value_loss / train_samples_count
        
        train_accuracy = 100 * correct_moves / total_moves
        top3_accuracy = 100 * top3_correct / total_moves
        top5_accuracy = 100 * top5_correct / total_moves
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_epoch_loss:.4f}, "
                   f"Train Accuracy: {train_accuracy:.2f}%, "
                   f"Top-3 Accuracy: {top3_accuracy:.2f}%, "
                   f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        
        if use_tensorboard:
            writer.add_scalar("Loss/train", train_epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/train_top3", top3_accuracy, epoch)
            writer.add_scalar("Accuracy/train_top5", top5_accuracy, epoch)
        
        model.eval()  
        val_loss = 0.0
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_samples_count = 0
        val_value_mse = 0.0
        
        val_top3_correct = 0
        val_top5_correct = 0
        
        value_predictions = []
        value_targets = []
        
        with torch.no_grad():
            for boards, targets, values, _ in val_loader:
                boards = boards.to(device)
                targets = targets.to(device)
                values = values.to(device)
                
                with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu'):
                    policy_logits, value_pred = model(boards)
                    loss, p_loss, v_loss = criterion(policy_logits, value_pred, targets, values)
                
                val_loss += loss.item() * boards.size(0)
                val_policy_loss += p_loss.item() * boards.size(0)
                val_value_loss += v_loss.item() * boards.size(0)
                val_samples_count += boards.size(0)
                
                pred_moves = torch.argmax(policy_logits, dim=1)
                true_moves = torch.argmax(targets, dim=1)
                val_correct += (pred_moves == true_moves).sum().item()
                
                val_value_mse += torch.mean((value_pred - values) ** 2).item() * boards.size(0)
                
                if len(value_predictions) < 5:
                    value_pred_detached = value_pred.detach().cpu().numpy()
                    values_detached = values.detach().cpu().numpy()
                    for vp, vt in zip(value_pred_detached, values_detached):
                        value_predictions.append(float(vp[0]))
                        value_targets.append(float(vt[0]))
                
                _, top_k_indices = torch.topk(policy_logits, k=5, dim=1)
                for i in range(len(true_moves)):
                    if true_moves[i] in top_k_indices[i, :3]:
                        val_top3_correct += 1
                    if true_moves[i] in top_k_indices[i, :5]:
                        val_top5_correct += 1
        
        if distributed:
            val_loss_tensor = torch.tensor([val_loss], device=device)
            val_policy_loss_tensor = torch.tensor([val_policy_loss], device=device)
            val_value_loss_tensor = torch.tensor([val_value_loss], device=device)
            val_correct_tensor = torch.tensor([val_correct], device=device)
            val_samples_tensor = torch.tensor([val_samples_count], device=device)
            val_value_mse_tensor = torch.tensor([val_value_mse], device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_policy_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_value_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_value_mse_tensor, op=dist.ReduceOp.SUM)
            
            val_loss = val_loss_tensor.item()
            val_policy_loss = val_policy_loss_tensor.item()
            val_value_loss = val_value_loss_tensor.item()
            val_correct = val_correct_tensor.item()
            val_samples_count = val_samples_tensor.item()
            val_value_mse = val_value_mse_tensor.item()
        
        val_epoch_loss = val_loss / val_samples_count
        val_epoch_policy_loss = val_policy_loss / val_samples_count
        val_epoch_value_loss = val_value_loss / val_samples_count
        val_accuracy = val_correct / val_samples_count * 100
        val_value_mse = val_value_mse / val_samples_count
        
        val_top3_accuracy = 100 * val_top3_correct / val_samples_count
        val_top5_accuracy = 100 * val_top5_correct / val_samples_count
        
        logger.info(f"Validation - "
                   f"Loss: {val_epoch_loss:.4f}, "
                   f"Accuracy: {val_accuracy:.2f}%, "
                   f"Top-3: {val_top3_accuracy:.2f}%, "
                   f"Top-5: {val_top5_accuracy:.2f}%")
        
        if use_tensorboard:
            writer.add_scalar("Loss/val", val_epoch_loss, epoch)
            writer.add_scalar("Accuracy/val", val_accuracy, epoch)
            writer.add_scalar("Accuracy/val_top3", val_top3_accuracy, epoch)
            writer.add_scalar("Accuracy/val_top5", val_top5_accuracy, epoch)
        
        epoch_time = time.time() - epoch_start_time
        
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr
            
        if rank == 0 and verbose:
            logger.info(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}")
        
        if not distributed or rank == 0:
            improvement = ""
            if epoch == 0:
                early_stopping = EarlyStopping(patience=patience, mode='max', verbose=not distributed or rank == 0)
            
            is_improved = early_stopping(val_accuracy)
            if is_improved:
                improvement = "  * "
                if not distributed:
                    best_model_state = model.state_dict().copy()
                else:
                    best_model_state = model.module.state_dict().copy()
                if not distributed:
                    torch.save(model.state_dict(), model_path)
                else:
                    torch.save(model.module.state_dict(), model_path)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
            print(f"{epoch+1:4d} | {train_epoch_loss:.6f} | {train_epoch_policy_loss:.6f} | {train_epoch_value_loss:.6f} | "
                  f"{val_epoch_loss:.6f} | {val_epoch_policy_loss:.6f} | {val_epoch_value_loss:.6f} | "
                  f"{val_accuracy:.2f}% {improvement}| {current_lr:.6f}")
            print(f"Accuracy - Train: {train_accuracy:.2f}%, Val: {val_accuracy:.2f}%, "
                  f"Top-3: {val_top3_accuracy:.2f}%, Top-5: {val_top5_accuracy:.2f}%")
            
            if value_predictions and value_targets:
                print("\nValue prediction samples (Pred | Target):")
                for pred, target in zip(value_predictions, value_targets):
                    error = abs(pred - target)
                    print(f"  {pred:+.4f} | {target:+.4f} (error: {error:.4f})")
                print("")  
            
            if local_logger:
                local_logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
                local_logger.info(f"Training - Loss: {train_epoch_loss:.6f}, Policy: {train_epoch_policy_loss:.6f}, Value: {train_epoch_value_loss:.6f}")
                local_logger.info(f"Validation - Loss: {val_epoch_loss:.6f}, Policy: {val_epoch_policy_loss:.6f}, Value: {val_epoch_value_loss:.6f}, Accuracy: {val_accuracy:.2f}%")
                local_logger.info(f"Learning rate: {current_lr:.6f}")
                
                if improvement:
                    local_logger.info(f"New best model saved with validation loss: {val_epoch_loss:.6f}")
                    
                if hasattr(local_logger, 'add_scalar'):
                    
                    local_logger.add_scalar('Loss/train', train_epoch_loss, epoch)
                    local_logger.add_scalar('Loss/val', val_epoch_loss, epoch)
                    
                    local_logger.add_scalar('PolicyLoss/train', train_epoch_policy_loss, epoch)
                    local_logger.add_scalar('PolicyLoss/val', val_epoch_policy_loss, epoch)
                    local_logger.add_scalar('ValueLoss/train', train_epoch_value_loss, epoch)
                    local_logger.add_scalar('ValueLoss/val', val_epoch_value_loss, epoch)
                    
                    local_logger.add_scalar('Accuracy/val', val_accuracy, epoch)
                    local_logger.add_scalar('ValueMSE/val', val_value_mse, epoch)
                    local_logger.add_scalar('LearningRate', current_lr, epoch)
            
            if visualizer:
                try:
                    visualizer.add_metrics(
                        epoch=epoch+1,
                        train_loss=train_epoch_loss,
                        val_loss=val_epoch_loss,
                        train_policy_loss=train_epoch_policy_loss,
                        val_policy_loss=val_epoch_policy_loss,
                        train_value_loss=train_epoch_value_loss,
                        val_value_loss=val_epoch_value_loss,
                        train_accuracy=train_accuracy/100,  
                        val_accuracy=val_accuracy/100,  
                        value_mse=val_value_mse,
                        learning_rate=current_lr
                    )
                    
                    visualizer.save_accuracy_plot_per_epoch()
                    visualizer.save_combined_metrics_per_epoch()
                    
                    if epoch == epochs - 1:
                        with torch.no_grad():
                            for i, (boards, targets, values, _) in enumerate(val_loader):
                                if i >= 3: 
                                    break
                                boards = boards.to(device)
                                targets = targets.to(device)
                                values = values.to(device)
                                
                                policy_logits, value_pred = model(boards)
                                if len(all_value_preds) < 10:
                                    for vp, vt in zip(value_pred[:5].cpu().numpy(), values[:5].cpu().numpy()):
                                        all_value_preds.append(float(vp[0]))
                                        all_value_targets.append(float(vt[0]))
                                if len(all_boards) < 1 and len(boards) > 0:
                                    all_boards.append(boards[0].cpu().numpy())
                                    all_policy_outputs.append(policy_logits[0].cpu().numpy())
                                
                                preds = torch.argmax(policy_logits, dim=1)
                                true = torch.argmax(targets, dim=1)
                                
                                if len(all_true_indices) < 20:
                                    all_true_indices.extend(true[:10].cpu().numpy())
                                    all_pred_indices.extend(preds[:10].cpu().numpy())
                        
                        if not all_true_indices:
                            all_true_indices = [0]
                            all_pred_indices = [0]
                        if not all_value_preds:
                            all_value_preds = [0.0]
                            all_value_targets = [0.0]
                        
                        safe_visualize(
                            visualizer,
                            value_preds=all_value_preds,
                            value_targets=all_value_targets,
                            top3_accuracy=top3_accuracies[-1] if top3_accuracies else None,
                            top5_accuracy=top5_accuracies[-1] if top5_accuracies else None
                        )
                        print(f"Training visualizations saved to: {visualizer.output_dir}/")
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Visualization skipped - visualizer not available")
        
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        top3_accuracies.append(top3_accuracy)
        top5_accuracies.append(top5_accuracy)
    
    if use_tensorboard:
        writer.close()
    
    if not distributed or rank == 0:
        if best_model_state is not None:
            torch.save(best_model_state, model_path)
            print(f"Best model (validation {'accuracy' if use_accuracy_metric else 'loss'}: "
                  f"{val_accuracy:.2f}%" if use_accuracy_metric else f"{val_epoch_loss:.6f}"
                  f") saved to {model_path}")
        else:
            if not distributed:
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model.module.state_dict(), model_path)
            print(f"Final model saved to {model_path}")
    
    return model, val_epoch_loss, val_accuracy

# --------------------------
# 7. Distributed Training Launcher
# --------------------------

def train_distributed(
    npz_path,
    model_path="gomoku_model.pth",
    load_model=None,
    board_size=15,
    validation_split=0.2,
    batch_size=64,
    epochs=10,
    lr=0.001,
    patience=5,
    policy_weight=1.0,
    value_weight=1.3,
    label_smoothing=0.1,
    use_stable_lr=False,
    verbose=True,
    visualize=True,
    num_gpus=None,
    random_seed=42
):
    """
    Distributed training function with multi-GPU support.
    Uses torch.distributed for data parallel training.
    
    Args:
        Same as train_model, plus:
        num_gpus (int, optional): Number of GPUs to use. Defaults to all available.
    
    Returns:
        The trained model.
    """
    num_available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = num_available_gpus
    
    if num_gpus <= 0:
        logger.error(f"No GPUs available for distributed training")
        logger.info(f"Falling back to single GPU/CPU training")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return train_model(npz_path, model_path, load_model, board_size, validation_split, batch_size, epochs, lr, patience, policy_weight, value_weight, label_smoothing, use_stable_lr, verbose, visualize, device)
    
    if num_gpus > num_available_gpus:
        logger.warning(f"Requested {num_gpus} GPUs but only {num_available_gpus} available")
        num_gpus = num_available_gpus
    
    logger.info(f"Starting distributed training with {num_gpus} GPUs")
    
    try:
        mp.spawn(
                _distributed_worker,
            args=(
                num_gpus,
                npz_path,
                model_path,
                load_model,
                board_size,
                validation_split,
                batch_size,
                epochs,
                lr,
                patience,
                policy_weight,
                value_weight,
                label_smoothing,
                use_stable_lr,
                verbose,
                visualize,
                random_seed
            ),
                nprocs=num_gpus,
                join=True
            )
        logger.info("Distributed training completed successfully")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = GomokuNet(board_size=board_size).to(device)
        model_path = model_path
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded best model from {model_path}")
        else:
            logger.warning(f"Could not find best model at {model_path}, returning un-trained model")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during distributed training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        logger.info("Attempting to fall back to single device training")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return train_model(npz_path, model_path, load_model, board_size, validation_split, batch_size, epochs, lr, patience, policy_weight, value_weight, label_smoothing, use_stable_lr, verbose, visualize, device)

def _distributed_worker(
    rank,
    world_size,
    npz_path,
    model_path,
    load_model,
    board_size,
    validation_split,
    batch_size,
    epochs,
    lr,
    patience,
    policy_weight,
    value_weight,
    label_smoothing,
    use_stable_lr,
    verbose,
    visualize,
    seed
):
    """
    Worker function for distributed training.
    
    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        Other arguments same as train_model.
    """
    setup_success = setup_distributed(rank, world_size)
    
    if not setup_success:
        logger.error(f"Failed to setup distributed environment for rank {rank}")
        return
    
    try:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        worker_logger = get_training_logger()
        worker_logger.info(f"Worker {rank}/{world_size-1} started on {device}")
        
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)

        worker_logger.info(f"Loading dataset from {npz_path}")
        full_dataset = GomokuDataset(npz_path)
        
        train_sampler = StratifiedSampler(full_dataset, validation_split=validation_split, is_validation=False, shuffle=True)
        val_sampler = StratifiedSampler(full_dataset, validation_split=validation_split, is_validation=True, shuffle=False)
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_sampler.indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_sampler.indices)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        
        num_workers = 0 if os.name == 'nt' else 4
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,  
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        worker_logger.info(f"Created data loaders with {len(train_loader)} training batches")       
        model = GomokuNet(board_size=board_size).to(device)
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=1e-4  
        )
        criterion = GomokuLoss(policy_weight=policy_weight, value_weight=value_weight,
                             label_smoothing=label_smoothing)
        cleanup_distributed()
        
    except Exception as e:
        worker_logger.error(f"Error in worker {rank}: {str(e)}")
        import traceback
        worker_logger.error(traceback.format_exc())
        
        try:
            cleanup_distributed()
        except:
            pass
        
        if rank != 0:  
            worker_logger.error(f"Worker {rank} exiting due to error")
            sys.exit(1)

def test_lr_scheduler(num_epochs=20, steps_per_epoch=100, learning_rate=1e-3):
    """
    Test the OneCycleLR scheduler by printing and plotting its values.
    
    Args:
        num_epochs (int): Number of epochs to simulate
        steps_per_epoch (int): Number of steps per epoch to simulate
        learning_rate (float): Maximum learning rate to use
    """
    import matplotlib.pyplot as plt
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3,  
        anneal_strategy='cos',  
        div_factor=25.0,  
        final_div_factor=10000.0,  
        three_phase=False  
    )
    
    learning_rates = []
    epochs = []
    
    print(f"OneCycleLR Scheduler Test")
    print(f"Max LR: {learning_rate}")
    print(f"Initial LR: {learning_rate/25.0:.6f}")
    print(f"Final LR: {learning_rate/10000.0:.6f}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup will peak at step: {int(total_steps * 0.3)}")
    
    step = 0
    for epoch in range(num_epochs):
        for i in range(steps_per_epoch):
            if i % (steps_per_epoch // 10) == 0 or i == steps_per_epoch - 1:
                current_step = epoch * steps_per_epoch + i
                current_epoch = epoch + i / steps_per_epoch
                epochs.append(current_epoch)
                current_lr = scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)
                print(f"Epoch {current_epoch:.2f} (Step {current_step}/{total_steps}), LR: {current_lr:.6f}")
            
            scheduler.step()
            step += 1
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, learning_rates, 'g-')
    plt.title('OneCycleLR Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('onecycle_lr_schedule.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning rate schedule plotted to onecycle_lr_schedule.png")
    print(f"Max learning rate: {max(learning_rates):.6f} at epoch ~{epochs[learning_rates.index(max(learning_rates))]:.2f}")
    print(f"Initial learning rate: {learning_rates[0]:.6f}")
    print(f"Final learning rate: {learning_rates[-1]:.6f}")

# -------------------
# 8. Run Training
# -------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a Gomoku policy-value network")
    parser.add_argument("--npz", type=str, required=True, help="Path to the NPZ file containing move data")
    parser.add_argument("--model_path", type=str, default="gomoku_model.pth", help="Path where the trained model will be saved")
    parser.add_argument("--load_model", type=str, default=None, help="Path to an existing model to continue training")
    parser.add_argument("--board_size", type=int, default=15, help="Size of the Gomoku board")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proportion of data to use for validation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--policy_weight", type=float, default=1.0, help="Weight for the policy loss")
    parser.add_argument("--value_weight", type=float, default=1.3, help="Weight for the value loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle", choices=["onecycle", "stable"],
                       help="Learning rate scheduler to use: 'onecycle' or 'stable' (constant)")
    parser.add_argument("--no_verbose", action="store_true", help="Disable verbose output")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization plots")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for distributed training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_scheduler", action="store_true", help="Test the learning rate scheduler")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Number of steps per epoch for scheduler testing")
    
    args = parser.parse_args()
    
    if args.test_scheduler:
        test_lr_scheduler(args.epochs, args.steps_per_epoch, args.lr)
    elif args.distributed:
        train_distributed(
            npz_path=args.npz,
            model_path=args.model_path,
            load_model=args.load_model,
            board_size=args.board_size,
            validation_split=args.val_split,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            label_smoothing=args.label_smoothing,
            use_stable_lr=(args.lr_scheduler == "stable"),
            verbose=not args.no_verbose,
            visualize=not args.no_visualize,
            num_gpus=args.num_gpus,
            random_seed=args.random_seed,
        )
    else:
        model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            npz_path=args.npz,
            model_path=args.model_path,
            load_model=args.load_model,
            board_size=args.board_size,
            validation_split=args.val_split,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            label_smoothing=args.label_smoothing,
            use_stable_lr=(args.lr_scheduler == "stable"),
            verbose=not args.no_verbose,
            visualize=not args.no_visualize,
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_seed=args.random_seed,
        )
