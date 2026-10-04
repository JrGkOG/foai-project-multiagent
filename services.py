"""
Micro-Service Workload Processes
==================================
Each service runs as a real Python process generating genuine load.
Memory is judged against a simulated per-container limit
(SERVICE_MEMORY_LIMIT_MB), the way Kubernetes compares usage to limits.
"""
import argparse
import time
import math
import os

from tools.process_metrics import SERVICE_MEMORY_LIMIT_MB


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
    Memory-intensive: fill ~88% of the simulated container limit, which lands
    above the 80% MEMORY-HIGH threshold without risking the host's RAM.
    """
    chunk = 1_000_000  # 1M list slots ≈ 8 MB per chunk
    target_bytes = int(SERVICE_MEMORY_LIMIT_MB * 1024 * 1024 * 0.88)
    num_chunks = max(1, target_bytes // (chunk * 8))

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

    print(f"[{args.service}] started mode={args.mode} pid={os.getpid()}", flush=True)
    MODE_MAP[args.mode]()
