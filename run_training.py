
import sys
import subprocess

# Execute the training command directly
cmd = ['python', 'main.py', 'train', '--npz', 'gomoku_dataset.npz', '--epochs', '20', '--batch_size', '64', '--lr', '0.001', '--smoothing', '0.1', '--policy_weight', '1.0', '--value_weight', '1.3', '--model_path', 'best_gomoku_model.pth', '--val_split', '0.1', '--patience', '5']
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
