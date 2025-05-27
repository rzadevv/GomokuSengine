import os
import psutil
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def monitor_process_memory(pid, interval=0.1, duration=None, log_file=None):
    """
    Monitor the memory usage of a process with the specified PID.
    
    Args:
        pid: Process ID to monitor
        interval: Sampling interval in seconds
        duration: How long to monitor in seconds (None for until process ends)
        log_file: File to save memory data
    
    Returns:
        Tuple containing (timestamps, memory_usage_mb, peak_memory_mb)
    """
    process = psutil.Process(pid)
    memory_usage = []
    timestamps = []
    start_time = time.time()
    
    print(f"Starting memory monitoring for PID {pid}")
    
    try:
        while True:
            # Get memory info in MB
            mem_info = process.memory_info()
            rss = mem_info.rss / (1024 * 1024)  # RSS in MB
            vms = mem_info.vms / (1024 * 1024)  # VMS in MB
            
            current_time = time.time() - start_time
            timestamps.append(current_time)
            memory_usage.append(rss)  # Using RSS as primary metric
            
            # Print current usage
            print(f"Time: {current_time:.2f}s, Memory (RSS): {rss:.2f} MB, (VMS): {vms:.2f} MB")
            
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"{current_time:.2f},{rss:.2f},{vms:.2f}\n")
            
            # Check if monitoring duration has elapsed
            if duration and (time.time() - start_time) >= duration:
                break
                
            # Sleep for the specified interval
            time.sleep(interval)
            
            # Check if process is still running
            if not psutil.pid_exists(pid) or process.status() == psutil.STATUS_ZOMBIE:
                print("Process has terminated")
                break
                
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Monitoring stopped: {e}")
    
    # Calculate peak memory
    peak_memory = max(memory_usage) if memory_usage else 0
    
    return timestamps, memory_usage, peak_memory

def run_gomoku_with_memory_profiling(args=None, interval=0.1, duration=None):
    """
    Run the Gomoku engine and monitor its memory usage.
    
    Args:
        args: Additional arguments to pass to the Gomoku process
        interval: Sampling interval for memory measurements
        duration: How long to monitor (None means until process ends)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"memory_profile_{timestamp}.csv"
    
    # Create and initialize the log file
    with open(log_file, 'w') as f:
        f.write("time,rss_mb,vms_mb\n")
    
    # Command to start the Gomoku engine
    cmd = ["python", "main.py"]
    if args:
        cmd.extend(args)
    
    print(f"Starting Gomoku engine with command: {' '.join(cmd)}")
    
    # Start the process
    process = subprocess.Popen(cmd)
    pid = process.pid
    
    print(f"Gomoku engine started with PID: {pid}")
    
    # Monitor memory
    timestamps, memory_usage, peak_memory = monitor_process_memory(
        pid, interval=interval, duration=duration, log_file=log_file
    )
    
    # Wait for the process to finish
    process.wait()
    
    # Generate report
    print("\nMemory Usage Summary:")
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Average memory usage: {np.mean(memory_usage):.2f} MB")
    print(f"Memory measurement count: {len(memory_usage)}")
    print(f"Detailed log saved to: {log_file}")
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage)
    plt.title("Gomoku Engine Memory Usage")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True)
    
    plot_file = f"memory_profile_{timestamp}.png"
    plt.savefig(plot_file)
    print(f"Memory usage plot saved to: {plot_file}")
    
    return {
        "peak_memory_mb": peak_memory,
        "avg_memory_mb": np.mean(memory_usage),
        "timestamps": timestamps,
        "memory_usage_mb": memory_usage,
        "log_file": log_file,
        "plot_file": plot_file
    }

def detailed_memory_snapshot():
    """Take a detailed snapshot of current memory usage by category"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_full_info()
    
    print("\nDetailed Memory Snapshot (MB):")
    print(f"RSS (Resident Set Size): {mem_info.rss / (1024 * 1024):.2f}")
    print(f"VMS (Virtual Memory Size): {mem_info.vms / (1024 * 1024):.2f}")
    
    # These may not be available on all platforms
    if hasattr(mem_info, 'uss'):
        print(f"USS (Unique Set Size): {mem_info.uss / (1024 * 1024):.2f}")
    if hasattr(mem_info, 'pss'):
        print(f"PSS (Proportional Set Size): {mem_info.pss / (1024 * 1024):.2f}")
    if hasattr(mem_info, 'swap'):
        print(f"Swap: {mem_info.swap / (1024 * 1024):.2f}")

if __name__ == "__main__":
    print("Gomoku Engine Memory Profiler")
    print("-----------------------------")
    
    # Get system information
    print(f"System memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # Run with 20-second time limit and 0.5s sampling interval
    results = run_gomoku_with_memory_profiling(interval=0.5, duration=20)
    
    # Show the plot (optional)
    plt.show() 