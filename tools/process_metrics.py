"""
Process Metrics Tool
=====================
Reads real CPU/memory from a running subprocess via psutil.
Derives request rate and P95 latency from the observed CPU/memory
(correlates realistically for the demo scenarios).
"""
from __future__ import annotations
import time
import logging
import psutil

logger = logging.getLogger(__name__)


def get_process_metrics(
    pid: int,
    service_name: str,
    replicas: int = 1,
    warm_up_secs: float = 1.5,
) -> dict:
    """Sample real CPU/memory of process `pid`."""
    proc = psutil.Process(pid)

    # Seed CPU counter, wait, then read
    proc.cpu_percent(interval=None)
    time.sleep(warm_up_secs)
    cpu_pct = min(proc.cpu_percent(interval=None), 100.0)

    mem_info = proc.memory_info()
    mem_pct  = proc.memory_percent()
    mem_mb   = mem_info.rss / (1024 ** 2)

    # Realistic derived metrics
    # CPU stress → high req rate + high latency
    # Memory stress → moderate req rate, moderate latency
    # Normal → low both
    if cpu_pct > 70:
        request_rate   = round(20 + cpu_pct * 1.1, 1)
        p95_latency_ms = round(60 + cpu_pct * 4.0, 1)
    elif mem_pct > 50:
        request_rate   = round(15 + mem_pct * 0.8, 1)
        p95_latency_ms = round(80 + mem_pct * 2.5, 1)
    else:
        request_rate   = round(5 + cpu_pct * 0.4, 1)
        p95_latency_ms = round(20 + cpu_pct * 1.2, 1)

    return {
        "service":        service_name,
        "cpu_percent":    round(cpu_pct, 2),
        "memory_percent": round(mem_pct, 2),
        "memory_mb":      round(mem_mb, 1),
        "replicas":       replicas,
        "request_rate":   request_rate,
        "p95_latency_ms": p95_latency_ms,
        "pid":            pid,
    }
