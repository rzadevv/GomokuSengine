import os
import sys
import time
import psutil
import argparse
import subprocess
import statistics
from datetime import datetime

def monitor_process_memory(pid, interval=0.5, max_time=20):
    """
    Monitor memory usage of a process with the given PID at regular intervals.
    
    Args:
        pid (int): Process ID to monitor
        interval (float): Time between measurements in seconds
        max_time (int): Maximum monitoring time in seconds
    """
    try:
        process = psutil.Process(pid)
        
        # Track memory usage over time
        start_time = time.time()
        memory_data = []
        peak_rss = 0
        
        print(f"Starting memory monitoring for PID {pid}")
        
        # Monitor until the process exits or max_time is reached
        while process.is_running() and (time.time() - start_time) < max_time:
            try:
                # Get memory info
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)  # RSS in MB
                vms_mb = mem_info.vms / (1024 * 1024)  # VMS in MB
                
                # Update peak memory
                peak_rss = max(peak_rss, rss_mb)
                
                # Store data
                elapsed = time.time() - start_time
                memory_data.append((elapsed, rss_mb, vms_mb))
                
                # Print current memory usage
                print(f"Time: {elapsed:.2f}s, Memory (RSS): {rss_mb:.2f} MB, (VMS): {vms_mb:.2f} MB")
                
                # Wait for the next interval
                time.sleep(interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
                
        print("Process has terminated\n")
        
        # Calculate summary statistics
        if memory_data:
            rss_values = [item[1] for item in memory_data]
            avg_rss = statistics.mean(rss_values)
            
            print("Memory Usage Summary:")
            print(f"Peak memory usage: {peak_rss:.2f} MB")
            print(f"Average memory usage: {avg_rss:.2f} MB")
            print(f"Memory measurement count: {len(memory_data)}")
        else:
            print("No memory measurements were collected")
            
    except psutil.NoSuchProcess:
        print(f"Error: Process with PID {pid} not found")
    except Exception as e:
        print(f"Error monitoring memory: {e}")
        import traceback
        traceback.print_exc()

def run_gomoku_engine(command, model_path=None):
    """
    Launch the Gomoku engine and monitor its memory usage.
    
    Args:
        command (str): Command to launch the Gomoku engine
        model_path (str, optional): Path to model file
    """
    # Get system information
    system_ram = psutil.virtual_memory().total / (1024**3)  # in GB
    print(f"System memory: {system_ram:.2f} GB")
    
    # Build the command
    cmd = command
    if "main.py" in command and model_path:
        # If running main.py and model path is provided, add model path argument
        cmd = f"{command} --model_path {model_path}"
    
    print(f"Starting Gomoku engine with command: {cmd}")
    
    try:
        # Start the process
        process = subprocess.Popen(cmd.split(), 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True,
                                  bufsize=1)
        
        # Get the process ID
        pid = process.pid
        print(f"Gomoku engine started with PID: {pid}")
        
        # Monitor the process memory usage
        monitor_process_memory(pid)
        
        # Wait for the process to complete and handle output
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
    except Exception as e:
        print(f"Error launching process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor memory usage of Gomoku engine")
    parser.add_argument("--command", type=str, default="python main.py infer", 
                        help="Command to launch the Gomoku engine")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to model file")
    parser.add_argument("--pid", type=int, default=None,
                        help="Directly monitor an existing process ID")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Monitoring interval in seconds")
    parser.add_argument("--max_time", type=int, default=20,
                        help="Maximum monitoring time in seconds")
    
    args = parser.parse_args()
    
    if args.pid:
        # Monitor an existing process
        monitor_process_memory(args.pid, args.interval, args.max_time)
    else:
        # Launch and monitor the engine
        run_gomoku_engine(args.command, args.model) 