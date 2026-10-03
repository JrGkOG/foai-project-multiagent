"""
Micro-Service Workload Processes
==================================
Each service runs as a real Python process generating genuine load.
"""
import argparse
import time
import math
import os
import sys
import psutil


def run_normal():
    """Light work: ~10-25% CPU, minimal memory."""
    while True:
        _ = sum(math.sqrt(i) for i in range(8_000))
        time.sleep(0.08)


def run_cpu_stress():
    """CPU-intensive: ~85-100% on one core, high latency."""
    while True:
        _ = sum(i * i for i in range(300_000))


def run_memory_stress():
    """
    Memory-intensive: allocate a significant fraction of system RAM.
    Targets ~60-80% of system memory to trigger the MEMORY bottleneck.
    """
    total_ram_bytes = psutil.virtual_memory().total
    # Try to hold 25% of total RAM in this single process
    target_bytes = int(total_ram_bytes * 0.25)
    chunk = 1_000_000  # 1M floats ≈ 8 MB per chunk
    num_chunks = target_bytes // (chunk * 8)
    num_chunks = max(10, min(num_chunks, 200))  # safety cap

    held = []
    for _ in range(num_chunks):
        held.append([1.0] * chunk)
        time.sleep(0.01)  # gradual allocation

    # Keep alive with light CPU
    while True:
        _ = sum(held[0][:500])
        time.sleep(0.3)


MODE_MAP = {
    "normal":        run_normal,
    "cpu_stress":    run_cpu_stress,
    "memory_stress": run_memory_stress,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="service")
    parser.add_argument("--mode", default="normal", choices=list(MODE_MAP.keys()))
    args = parser.parse_args()

    pid_file = f"/tmp/advisor_{args.service}.pid"
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    print(f"[{args.service}] started mode={args.mode} pid={os.getpid()}", flush=True)
    MODE_MAP[args.mode]()
