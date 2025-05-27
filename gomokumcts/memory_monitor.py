#!/usr/bin/env python
import os
import psutil
import subprocess
import time
import signal
import sys
import argparse
import numpy as np

def get_system_memory():
    """Get total system memory in GB"""
    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3)

def start_engine(command):
    """Start the Gomoku engine process"""
    print(f"Starting Gomoku engine with command: {command}")
    # Don't capture output to avoid blocking or complexity
    process = subprocess.Popen(command.split())
    print(f"Gomoku engine started with PID: {process.pid}")
    return process

def monitor_memory(pid, interval=0.5, max_runtime=20):
    """Monitor memory usage of a process with the given PID"""
    try:
        process = psutil.Process(pid)
        memory_usage = []
        start_time = time.time()
        
        print(f"Starting memory monitoring for PID {pid}")
        print(f"Monitoring will stop after {max_runtime} seconds")
        
        while process.is_running():
            # Get memory info
            try:
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)  # RSS in MB
                vms_mb = mem_info.vms / (1024 * 1024)  # VMS in MB
                
                elapsed = time.time() - start_time
                print(f"Time: {elapsed:.2f}s, Memory (RSS): {rss_mb:.2f} MB, (VMS): {vms_mb:.2f} MB")
                
                memory_usage.append(rss_mb)
                
                # Check if we've reached the time limit
                if elapsed >= max_runtime:
                    print(f"Reached maximum runtime of {max_runtime} seconds")
                    if process.is_running():
                        print(f"Terminating process {pid}")
                        process.terminate()
                    break
                
                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("Process has terminated or access denied")
                break
        
        print("Process has terminated")
        
        # Calculate memory statistics
        if memory_usage:
            peak_memory = max(memory_usage)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            print("\nMemory Usage Summary:")
            print(f"Peak memory usage: {peak_memory:.2f} MB")
            print(f"Average memory usage: {avg_memory:.2f} MB")
            print(f"Memory measurement count: {len(memory_usage)}")
    
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}")
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        if process.is_running():
            process.terminate()
            print(f"Process {pid} terminated")

def main():
    parser = argparse.ArgumentParser(description='Monitor memory usage of Gomoku MCTS engine')
    parser.add_argument('--command', type=str, default='python mcts_demo.py',
                        help='Command to start the Gomoku engine')
    parser.add_argument('--pid', type=int, help='Monitor an already running process with this PID')
    parser.add_argument('--interval', type=float, default=0.5, 
                        help='Interval in seconds between measurements')
    parser.add_argument('--timeout', type=float, default=20.0,
                        help='Maximum runtime in seconds (default: 20)')
    
    args = parser.parse_args()
    
    # Print system memory
    system_memory = get_system_memory()
    print(f"System memory: {system_memory:.2f} GB")
    
    # Start monitoring
    if args.pid:
        monitor_memory(args.pid, args.interval, args.timeout)
    else:
        process = start_engine(args.command)
        monitor_memory(process.pid, args.interval, args.timeout)

if __name__ == "__main__":
    main() 