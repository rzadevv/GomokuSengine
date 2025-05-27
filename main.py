import argparse
import torch
import numpy as np
import os
import sys
import traceback
import time
from logger import get_evaluation_logger, get_training_logger, get_inference_logger

# ----------------------------------------
# PyTorch Memory Optimizations
# ----------------------------------------
import gc
torch.backends.cudnn.benchmark = True  # Speed up consistent workloads
torch.set_num_threads(2)  # Limit CPU threads for better memory efficiency
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False  # More memory efficient
    torch.backends.cuda.matmul.allow_tf32 = True  # Better performance on newer GPUs
    
# Set environment variables to limit background threads
os.environ['OMP_NUM_THREADS'] = '2'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '2'  # MKL threads
    
print("Applied PyTorch memory optimizations")

def main():
    parser = argparse.ArgumentParser(description="Gomoku Engine Main Script")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train, eval, infer, gui, train_gui, accuracy, visualize")

    # ----------------------
    # Data pipeline mode
    # ----------------------
    data_parser = subparsers.add_parser("data", help="Process PSQ files and generate dataset")
    data_parser.add_argument("--psq_dir", type=str, default="psq_games", help="Directory containing PSQ files")
    data_parser.add_argument("--output", type=str, default="gomoku_dataset.npz", help="Output dataset file path")
    data_parser.add_argument("--board_size", type=int, default=15, help="Board size to generate (default 15x15)")
    
    # ----------------------
    # Train mode arguments
    # ----------------------
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--npz", type=str, default="gomoku_dataset.npz", help="Path to the dataset .npz file")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing factor")
    train_parser.add_argument("--policy_weight", type=float, default=1.0, help="Weight for policy loss component")
    train_parser.add_argument("--value_weight", type=float, default=1.3, help="Weight for value loss component")
    train_parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                              help="Path to save the model checkpoint")
    train_parser.add_argument("--lr_scheduler", type=str, default="onecycle", choices=["onecycle", "stable"],
                              help="Learning rate scheduler to use: 'onecycle' or 'stable' (constant)")
    train_parser.add_argument("--distributed", action="store_true", help="Enable distributed multi-GPU training")
    train_parser.add_argument("--num_gpus", type=int, default=None, 
                              help="Number of GPUs to use for distributed training (defaults to all available)")
    train_parser.add_argument("--val_split", type=float, default=0.1,
                              help="Fraction of data to use for validation (default: 0.1)")
    train_parser.add_argument("--patience", type=int, default=5,
                              help="Early stopping patience - stop if validation loss doesn't improve (default: 5)")
    train_parser.add_argument("--stable_lr", action="store_true", 
                              help="Use a stable learning rate without scheduling")
    
    # ----------------------
    # Training GUI mode
    # ----------------------
    train_gui_parser = subparsers.add_parser("train_gui", help="Launch the Training GUI")
    train_gui_parser.add_argument("--dataset", type=str, default="gomoku_dataset.npz", 
                                help="Path to the dataset file to pre-populate in the GUI")
    
    # ----------------------
    # Evaluation mode arguments
    # ----------------------
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--npz", type=str, default="gomoku_dataset.npz", help="Path to the evaluation dataset .npz file")
    eval_parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                             help="Path to the trained model checkpoint")
    
    # ----------------------
    # Inference mode arguments
    # ----------------------
    infer_parser = subparsers.add_parser("infer", help="Run inference on a sample board")
    infer_parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                              help="Path to the trained model checkpoint")
    infer_parser.add_argument("--temperature", type=float, default=1.0,
                              help="Temperature scaling factor for inference")
    infer_parser.add_argument("--visualize", action="store_true",
                              help="Visualize the board and prediction")
    
    # ----------------------
    # GUI mode arguments
    # ----------------------
    gui_parser = subparsers.add_parser("gui", help="Launch the Tkinter GUI for Human vs AI play")
    gui_parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                           help="Path to the model to use")
    
    # ----------------------
    # New Accuracy Evaluation mode
    # ----------------------
    accuracy_parser = subparsers.add_parser("accuracy", help="Evaluate model accuracy on test data")
    accuracy_parser.add_argument("--npz", type=str, default="gomoku_dataset.npz", 
                               help="Path to the dataset .npz file")
    accuracy_parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                               help="Path to the trained model checkpoint")
    accuracy_parser.add_argument("--batch_size", type=int, default=64,
                               help="Batch size for evaluation")
    accuracy_parser.add_argument("--test_split", type=float, default=0.1,
                               help="Fraction of data to use for testing (default: 0.1)")
    accuracy_parser.add_argument("--visualize", action="store_true",
                               help="Visualize examples where the model prediction is correct/incorrect")
    accuracy_parser.add_argument("--temperature", type=float, default=1.0,
                               help="Temperature for softmax scaling")
    
    # ----------------------
    # New Visualization Mode for model accuracy
    # ----------------------
    visualize_parser = subparsers.add_parser("visualize", help="Generate visualizations of model accuracy and performance")
    visualize_parser.add_argument("--model_path", type=str, default="best_gomoku_model.pth",
                                 help="Path to the trained model checkpoint")
    visualize_parser.add_argument("--npz", type=str, default="gomoku_dataset.npz", 
                                 help="Path to the dataset .npz file")
    visualize_parser.add_argument("--batch_size", type=int, default=64,
                                 help="Batch size for data processing")
    visualize_parser.add_argument("--test_split", type=float, default=0.1,
                                 help="Fraction of data to use for testing")
    visualize_parser.add_argument("--output_dir", type=str, default="accuracy_visualization",
                                 help="Directory to save visualizations")
    visualize_parser.add_argument("--track_training", action="store_true",
                                 help="Load training data from logs instead of computing from scratch")
    
    args = parser.parse_args()
    
    if args.mode == "data":
        try:
            from datapipeline import process_files, save_dataset
            import glob
            
            # Find all PSQ files in the specified directory (recursively)
            psq_files = []
            for root, dirs, files in os.walk(args.psq_dir):
                for file in files:
                    if file.endswith(".psq"):
                        psq_files.append(os.path.join(root, file))
            
            if not psq_files:
                print(f"Error: No PSQ files found in {args.psq_dir}")
                return
                
            print(f"Found {len(psq_files)} PSQ files")
            
            # Process the files
            boards, targets, phases = process_files(psq_files, output_size=args.board_size)
            print(f"Generated {len(boards)} training examples (including augmentations)")
            
            # Save the dataset
            save_dataset(args.output, boards, targets, phases)
            print(f"Dataset saved to {args.output}")
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return
    
    elif args.mode == "train":
        try:
            from train import train_model, train_distributed
            
            if not os.path.exists(args.npz):
                print(f"Error: Dataset file {args.npz} not found.")
                return
            
            start_time = time.time()
            print(f"Starting training with {args.epochs} epochs...")
            print(f"Policy loss weight: {args.policy_weight}, Value loss weight: {args.value_weight}")
            
            if args.distributed:
                # Initialize distributed training
                print(f"Using distributed training with {args.num_gpus} GPUs")
                model = train_distributed(
                    npz_path=args.npz,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    label_smoothing=args.smoothing,
                    policy_weight=args.policy_weight, 
                    value_weight=args.value_weight,
                    num_gpus=args.num_gpus,
                    validation_split=args.val_split,
                    patience=args.patience,
                    use_stable_lr=args.stable_lr
                )
            else:
                # Single GPU or CPU training
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device}")
                
                model = train_model(
                    npz_path=args.npz,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    label_smoothing=args.smoothing,
                    policy_weight=args.policy_weight, 
                    value_weight=args.value_weight,
                    device=device,
                    validation_split=args.val_split,
                    patience=args.patience,
                    use_stable_lr=args.stable_lr
                )
            
            elapsed_time = time.time() - start_time
            print(f"Training completed in {elapsed_time/60:.2f} minutes")
            
            # Explicit garbage collection after training
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback as tb
            tb.print_exc()
            return
    
    elif args.mode == "eval":
        try:
            # Initialize logger
            logger = get_evaluation_logger()
            logger.info(f"Starting evaluation with model: {args.model_path}")
            
            # Check if files exist
            if not os.path.exists(args.npz):
                logger.error(f"Dataset file {args.npz} not found")
                print(f"Error: Dataset file {args.npz} not found")
                return
                
            if not os.path.exists(args.model_path):
                logger.error(f"Model file {args.model_path} not found")
                print(f"Error: Model file {args.model_path} not found")
                return
                
            # Simple evaluation: compute average loss and accuracy over the dataset.
            from train import GomokuDataset
            from model import GomokuNet
            import torch.nn as nn

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            logger.reset_timer()  # Start timing
            dataset = GomokuDataset(args.npz)
            # Use 0 workers on Windows to avoid pickling issues
            num_workers = 0 if os.name == 'nt' else 4
            logger.info(f"Using {num_workers} workers for data loading")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, 
                                                    num_workers=num_workers, pin_memory=True)
            logger.info(f"Dataset loaded: {len(dataset)} examples")
            
            model = GomokuNet().to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            logger.info("Model loaded successfully")
            
            # For evaluation, we'll use KLDivLoss since targets are one-hot.
            criterion = nn.KLDivLoss(reduction='batchmean')
            total_loss = 0.0
            total_samples = 0
            correct = 0
            
            with torch.no_grad():
                for i, (boards, targets, _) in enumerate(dataloader):
                    if i % 10 == 0:
                        logger.info(f"Processing batch {i}/{len(dataloader)}")
                        # Add garbage collection every 10 batches
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    boards = boards.to(device)
                    # Note: targets are already flattened in the dataset
                    targets = targets.to(device)
                    logits = model(boards)  # (B, 225)
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, targets)
                    total_loss += loss.item() * boards.size(0)
                    total_samples += boards.size(0)
                    
                    # Compute accuracy: compare the index with max probability
                    preds = torch.argmax(logits, dim=1)
                    true = torch.argmax(targets, dim=1)
                    correct += (preds == true).sum().item()
                    
                    # Clean up GPU memory after each batch
                    del boards, targets, logits, log_probs, loss, preds, true
            
            avg_loss = total_loss / total_samples
            accuracy = correct / total_samples * 100
            
            # Log the metrics
            logger.evaluation_metrics({
                "loss": avg_loss,
                "accuracy": accuracy,
                "total_samples": total_samples
            })
            
            logger.log_elapsed_time("Evaluation")
            print(f"Evaluation Loss: {avg_loss:.4f}")
            print(f"Evaluation Accuracy: {accuracy:.2f}%")
            
            # Final garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            if 'logger' in locals():
                logger.error(f"Error during evaluation: {e}")
            print(f"Error during evaluation: {e}")
            return
    
    elif args.mode == "infer":
        try:
            # Initialize logger
            logger = get_inference_logger()
            logger.info(f"Starting inference with model: {args.model_path}")
            logger.info(f"Temperature: {args.temperature}, Visualize: {args.visualize}")
            
            # Check if model file exists
            if not os.path.exists(args.model_path):
                logger.error(f"Model file {args.model_path} not found")
                print(f"Error: Model file {args.model_path} not found")
                return
                
            from inference import load_model, predict_move_with_details, visualize_board_and_prediction, place_stone
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            logger.reset_timer()  # Start timing
            model = load_model(args.model_path, device=device)
            logger.info("Model loaded successfully")
            
            # Garbage collect after model loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create a sample raw board: empty board where channel 2 is 1 everywhere.
            raw_board = np.zeros((15, 15, 3), dtype=np.float32)
            raw_board[:, :, 2] = 1.0
            
            # Add some example stones for a more interesting demonstration
            # Example: Create an early midgame position with several stones
            # Black stones
            black_positions = [(8, 8), (9, 9), (10, 10), (7, 7), (6, 6)]
            for row, col in black_positions:
                raw_board[row-1, col-1, 0] = 1.0  # Black stone (1-indexed to 0-indexed)
                raw_board[row-1, col-1, 2] = 0.0  # No longer empty
                
            # White stones
            white_positions = [(8, 9), (9, 8), (7, 8), (6, 7)]
            for row, col in white_positions:
                raw_board[row-1, col-1, 1] = 1.0  # White stone (1-indexed to 0-indexed)
                raw_board[row-1, col-1, 2] = 0.0  # No longer empty
                
            logger.info("Created sample board with multiple stones")
            
            # Set up value evaluation parameters
            top_k = 5  # Number of top policy moves to evaluate with value head
            use_value_eval = True  # Enable Method 2 (value evaluation)
            
            logger.info(f"Using value-based evaluation (Method 2) with top_k={top_k}")
            logger.info("Predicting move...")
            
            # Use the detailed prediction function
            top_move, probs, value, evaluated_moves = predict_move_with_details(
                model, 
                raw_board, 
                temperature=args.temperature, 
                device=device,
                top_k=top_k,
                use_value_eval=use_value_eval
            )
            
            logger.info(f"Predicted move (1-indexed): {top_move}")
            logger.info(f"Initial position evaluation: {value:.4f}")
            
            # Calculate confidence of prediction from policy
            confidence = float(probs[top_move[0]-1, top_move[1]-1])
            logger.info(f"Policy confidence: {confidence:.4f}")
            
            # Log details of evaluated moves
            if evaluated_moves:
                logger.info("Value evaluation details:")
                # Sort by 'our_value' (best for us first)
                sorted_moves = sorted(evaluated_moves, key=lambda x: x['our_value'], reverse=True)
                
                for i, move_data in enumerate(sorted_moves):
                    move = move_data['move']
                    policy_prob = move_data['policy_prob']
                    our_value = move_data['our_value']
                    next_value = move_data['next_value']
                    
                    logger.info(f"  #{i+1}: Move {move} - Policy: {policy_prob:.4f}, " +
                               f"Next Value: {next_value:.4f}, Our Expected Value: {our_value:.4f}")
                
                # Highlight if the chosen move wasn't the top policy move
                top_policy_move = np.unravel_index(np.argmax(probs), probs.shape)
                top_policy_move = (top_policy_move[0] + 1, top_policy_move[1] + 1)  # Convert to 1-indexed
                
                if top_move != top_policy_move:
                    logger.info(f"Note: Value-based selection chose {top_move} over " +
                               f"top policy move {top_policy_move}")
            
            # Show what happens if we actually make this move
            if evaluated_moves:
                try:
                    # Find our chosen move in the evaluated moves
                    for move_data in evaluated_moves:
                        if move_data['move'] == top_move:
                            # Determine current player
                            black_stones = np.sum(raw_board[:, :, 0])
                            white_stones = np.sum(raw_board[:, :, 1])
                            current_player = black_stones <= white_stones  # True if black's turn
                            player_name = "Black" if current_player else "White"
                            
                            logger.info(f"After {player_name} plays at {top_move}, position value will be {move_data['next_value']:.4f}")
                            logger.info(f"This value is from {player_name}'s opponent's perspective")
                            break
                except Exception as e:
                    logger.error(f"Error analyzing chosen move: {e}")
            
            if args.visualize:
                try:
                    logger.info("Generating visualization...")
                    visualize_board_and_prediction(raw_board, probs, top_move, value, evaluated_moves)
                except Exception as e:
                    logger.error(f"Error in visualization: {e}")
                    
            logger.log_elapsed_time("Inference")
            print("Predicted move (1-indexed):", top_move)
            
            # Garbage collect after inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                    
        except Exception as e:
            if 'logger' in locals():
                logger.error(f"Error during inference: {e}")
            print(f"Error during inference: {e}")
            return
    
    elif args.mode == "gui":
        try:
            # Launch the GUI
            from gui import GomokuGUI
            import tkinter as tk
            
            # Override the default model path if provided
            if args.model_path != "best_gomoku_model.pth":
                # Update the MODEL_PATH constant in gui module
                import gui
                gui.MODEL_PATH = args.model_path
                
            root = tk.Tk()
            app = GomokuGUI(root)
            
            # Run garbage collection periodically during GUI operation
            def periodic_gc():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Schedule the next garbage collection
                root.after(10000, periodic_gc)  # Run every 10 seconds
                
            # Start periodic garbage collection
            root.after(10000, periodic_gc)
            
            root.mainloop()
            
        except Exception as e:
            print(f"Error launching GUI: {e}")
            return
    
    elif args.mode == "train_gui":
        try:
            # Launch the training GUI
            from train_gui import TrainingGUI
            import tkinter as tk
            
            root = tk.Tk()
            app = TrainingGUI(root)
            
            # Set initial dataset if provided
            if args.dataset:
                app.dataset_var.set(args.dataset)
                
            # Run garbage collection periodically during GUI operation
            def periodic_gc():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Schedule the next garbage collection
                root.after(10000, periodic_gc)  # Run every 10 seconds
                
            # Start periodic garbage collection
            root.after(10000, periodic_gc)
            
            root.mainloop()
            
        except Exception as e:
            print(f"Error launching Training GUI: {e}")
            return
    
    elif args.mode == "accuracy":
        try:
            from train import GomokuDataset, StratifiedSampler
            import torch.nn as nn
            from model import GomokuNet
            from torch.utils.data import DataLoader, Subset
            
            # Initialize logger
            logger = get_evaluation_logger()
            logger.info(f"Starting accuracy test for model: {args.model_path}")
            
            # Check if files exist
            if not os.path.exists(args.npz):
                logger.error(f"Dataset file {args.npz} not found")
                print(f"Error: Dataset file {args.npz} not found")
                return
                
            if not os.path.exists(args.model_path):
                logger.error(f"Model file {args.model_path} not found")
                print(f"Error: Model file {args.model_path} not found")
                return
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            model = GomokuNet().to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            # Don't switch to eval mode to keep dropout active during evaluation
            # model.eval()
            
            logger.info("Model loaded successfully")
            
            # Garbage collection after model loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load dataset
            logger.info(f"Loading dataset from {args.npz}")
            full_dataset = GomokuDataset(args.npz)
            
            # Split dataset for testing
            dataset_size = len(full_dataset)
            test_size = int(dataset_size * args.test_split)
            train_size = dataset_size - test_size
            _, test_dataset = random_split(full_dataset, [train_size, test_size])
            
            logger.info(f"Using {test_size} examples for accuracy evaluation")
            
            # Create test dataloader
            num_workers = 0 if os.name == 'nt' else 4
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
            # Tracking metrics
            total = 0
            top1_correct = 0
            top3_correct = 0
            top5_correct = 0
            
            # Lists to track phase-specific accuracy
            phases = ['opening', 'midgame', 'endgame']
            phase_counts = {phase: 0 for phase in phases}
            phase_correct = {phase: 0 for phase in phases}
            
            # For visualization
            correct_examples = []
            incorrect_examples = []
            
            # Run evaluation
            with torch.no_grad():
                for i, (boards, targets, values, phase_batch) in enumerate(test_loader):
                    if i % 5 == 0:  # Perform GC every 5 batches
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    boards = boards.to(device)
                    targets = targets.to(device)
                    
                    # Apply temperature scaling
                    policy_logits, _ = model(boards)
                    scaled_logits = policy_logits / args.temperature
                    probs = torch.softmax(scaled_logits, dim=1)
                    
                    # Get top-k predictions
                    pred_moves = torch.argmax(policy_logits, dim=1)
                    true_moves = torch.argmax(targets, dim=1)
                    
                    # Calculate top-1 accuracy
                    correct = (pred_moves == true_moves).cpu().numpy()
                    top1_correct += np.sum(correct)
                    
                    # Calculate top-k accuracy
                    _, top_indices = torch.topk(policy_logits, k=5, dim=1)
                    
                    for j in range(len(true_moves)):
                        # Add to total count
                        total += 1
                        
                        # Get the phase for this example
                        curr_phase = phase_batch[j]
                        phase_counts[curr_phase] += 1
                        
                        # Check if correct prediction
                        if correct[j]:
                            phase_correct[curr_phase] += 1
                        
                        # Check top-3 and top-5
                        if true_moves[j] in top_indices[j, :3]:
                            top3_correct += 1
                        if true_moves[j] in top_indices[j, :5]:
                            top5_correct += 1
                        
                        # Save examples for visualization (limit to 10 each)
                        if args.visualize:
                            board_np = boards[j].cpu().numpy()
                            target_np = targets[j].cpu().numpy()
                            prob_np = probs[j].cpu().numpy()
                            
                            example = (board_np, target_np, prob_np, curr_phase)
                            
                            if correct[j] and len(correct_examples) < 10:
                                correct_examples.append(example)
                            elif not correct[j] and len(incorrect_examples) < 10:
                                incorrect_examples.append(example)
                                
                    # Clean up tensors
                    del boards, targets, policy_logits, scaled_logits, probs, pred_moves, true_moves, top_indices
            
            # Calculate accuracy metrics
            top1_accuracy = top1_correct / total * 100
            top3_accuracy = top3_correct / total * 100
            top5_accuracy = top5_correct / total * 100
            
            # Calculate phase-specific accuracy
            phase_accuracy = {phase: phase_correct[phase] / max(1, phase_counts[phase]) * 100 for phase in phases}
            
            # Log results
            logger.info(f"Evaluation complete on {total} examples")
            logger.info(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total})")
            logger.info(f"Top-3 Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{total})")
            logger.info(f"Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total})")
            
            for phase in phases:
                logger.info(f"{phase.capitalize()} Accuracy: {phase_accuracy[phase]:.2f}% ({phase_correct[phase]}/{phase_counts[phase]})")
            
            # Print results to console
            print(f"\nAccuracy Evaluation Results:")
            print(f"-----------------------------")
            print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total})")
            print(f"Top-3 Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{total})")
            print(f"Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total})")
            print(f"\nPhase-Specific Accuracy:")
            for phase in phases:
                print(f"- {phase.capitalize()}: {phase_accuracy[phase]:.2f}% ({phase_correct[phase]}/{phase_counts[phase]})")
            
            # Final garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Visualize examples if requested
            if args.visualize and (correct_examples or incorrect_examples):
                from visualization import visualize_prediction_examples
                visualize_prediction_examples(correct_examples, incorrect_examples)
                
        except Exception as e:
            print(f"Error during accuracy evaluation: {e}")
            traceback.print_exc()
            return
    
    elif args.mode == "visualize":
        try:
            from visualization import visualize_accuracy_metrics, collect_accuracy_data
            from train import GomokuDataset
            from model import GomokuNet
            from torch.utils.data import DataLoader, random_split
            
            # Initialize logger
            logger = get_evaluation_logger()
            logger.info(f"Starting visualization for model: {args.model_path}")
            
            # Check if files exist
            if not os.path.exists(args.npz):
                logger.error(f"Dataset file {args.npz} not found")
                print(f"Error: Dataset file {args.npz} not found")
                return
                
            if not os.path.exists(args.model_path):
                logger.error(f"Model file {args.model_path} not found")
                print(f"Error: Model file {args.model_path} not found")
                return
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            model = GomokuNet().to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            logger.info("Model loaded successfully")
            
            # Garbage collection after model loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load dataset and create data loaders
            logger.info("Loading dataset...")
            full_dataset = GomokuDataset(args.npz)
            
            # Split dataset for testing
            dataset_size = len(full_dataset)
            test_size = int(dataset_size * args.test_split)
            train_size = dataset_size - test_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
            
            logger.info(f"Dataset split: {train_size} training samples, {test_size} test samples")
            
            # Create data loaders
            num_workers = 0 if os.name == 'nt' else 4
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=num_workers
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
            # Collect accuracy data
            logger.info("Collecting accuracy metrics...")
            
            if args.track_training:
                # Load training metrics from logs (if available)
                # This is a placeholder - implement based on your logging system
                try:
                    import json
                    with open('logs/training_metrics.json', 'r') as f:
                        accuracy_data = json.load(f)
                    logger.info("Loaded training history from logs")
                except:
                    logger.warning("Could not load training history, computing metrics from scratch")
                    game_phases = ['opening', 'midgame', 'endgame']
                    accuracy_data = collect_accuracy_data(model, train_loader, test_loader, device, game_phases)
            else:
                # Compute metrics from scratch
                game_phases = ['opening', 'midgame', 'endgame']
                accuracy_data = collect_accuracy_data(model, train_loader, test_loader, device, game_phases)
                
                # Run garbage collection after collecting data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            report_path = visualize_accuracy_metrics(accuracy_data, output_dir=args.output_dir)
            
            logger.info(f"Visualization complete! Report saved to: {report_path}")
            print(f"\nAccuracy visualizations generated in {args.output_dir}/")
            print(f"Open {report_path} in a web browser to view the report")
            
            # Final garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return
    
    else:
        parser.print_help()
        
    # Final garbage collection at the end of any mode
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
