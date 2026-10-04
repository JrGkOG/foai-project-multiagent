#!/usr/bin/env python3
"""
Multi-Agent Container Resource Advisor – CLI Entry Point
=========================================================
Usage:
    python main.py --scenario normal
    python main.py --scenario cpu_stress
    python main.py --scenario memory_stress
    python main.py --all
    python main.py --all --no-llm     (rule-based only, no API key needed)
    python main.py            (interactive menu)
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.markup import escape

from tools.process_metrics import get_process_metrics
from pipeline import run_pipeline
from agents.llm_factory import disable_llm, llm_mode_label, llm_snapshot

# ── Suppress noisy library warnings in CLI output ────────────────────────────
logging.basicConfig(level=logging.ERROR)
for noisy in ("agents.monitoring_agent", "agents.bottleneck_agent",
              "agents.scaling_agent", "agents.recommendation_agent",
              "agents.llm_factory"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

console = Console()
PROJECT_ROOT = Path(__file__).parent
# Starting replica count per service (worker > 2 so scale-down is reachable)
SERVICE_REPLICAS = {"backend": 2, "frontend": 2, "worker": 3}
SERVICES     = list(SERVICE_REPLICAS)
WARMUP_SECS  = 3.5
SAMPLE_SECS  = 2.0

# ── Colour helpers ───────────────────────────────────────────────────────────

def _color_pct(value: float, warn: float = 60, crit: float = 80) -> Text:
    label = f"{value:.1f}%"
    if value >= crit:  return Text(label, style="bold red")
    if value >= warn:  return Text(label, style="bold yellow")
    return Text(label, style="bold green")

def _action_style(action: str) -> Text:
    styles = {
        "SCALE_UP":   ("⬆  SCALE UP",   "bold red"),
        "SCALE_DOWN": ("⬇  SCALE DOWN", "bold cyan"),
        "NO_ACTION":  ("✔  NO ACTION",  "bold green"),
    }
    label, style = styles.get(action, (action, "white"))
    return Text(label, style=style)

def _severity_style(sev: str) -> Text:
    s = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "green", "NONE": "dim"}.get(sev, "white")
    return Text(sev, style=s)

def _bar(value: float) -> str:
    filled = int(value * 20)
    return "█" * filled + "░" * (20 - filled)

# ── Process lifecycle ────────────────────────────────────────────────────────

def spawn_services(scenario: str, procs: dict[str, subprocess.Popen]) -> None:
    """Start every service, registering each in `procs` as soon as it exists
    so the caller can always clean up, even if a later spawn fails."""
    for svc in SERVICES:
        procs[svc] = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "services.py"),
             "--service", svc, "--mode", scenario],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def collect_all_metrics(procs: dict[str, subprocess.Popen]) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for svc, proc in procs.items():
        if proc.poll() is not None:
            console.print(f"[red]{svc} exited early (code {proc.returncode}); "
                          f"skipping.[/red]")
            continue
        try:
            m = get_process_metrics(pid=proc.pid, service_name=svc,
                                    replicas=SERVICE_REPLICAS[svc],
                                    warm_up_secs=SAMPLE_SECS)
            metrics[svc] = m
        except Exception as e:
            console.print(f"[red]Failed to measure {svc}: {e}[/red]")
    return metrics

def kill_services(procs: dict[str, subprocess.Popen]) -> None:
    """Terminate every service and reap it (no zombies, no orphans)."""
    for p in procs.values():
        try: p.terminate()
        except Exception: pass
    for p in procs.values():
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
                p.wait(timeout=3)
            except Exception: pass
        except Exception: pass

# ── Display ──────────────────────────────────────────────────────────────────

def print_metrics_table(all_metrics: dict[str, dict]) -> None:
    t = Table(title="📊 Live Process Metrics  (real psutil data)",
              box=box.ROUNDED, border_style="blue")
    t.add_column("Service",   style="bold white", no_wrap=True)
    t.add_column("CPU %",     justify="right")
    t.add_column("Mem %",     justify="right")
    t.add_column("Mem (MB)",  justify="right")
    t.add_column("Replicas",  justify="center")
    t.add_column("Req/s",     justify="right")
    t.add_column("P95 Lat",   justify="right")
    for svc, m in all_metrics.items():
        t.add_row(
            svc,
            _color_pct(m["cpu_percent"]),
            _color_pct(m["memory_percent"], 60, 80),
            f"{m['memory_mb']:.0f}",
            str(m["replicas"]),
            f"{m['request_rate']:.1f}",
            f"{m['p95_latency_ms']:.0f} ms",
        )
    console.print(t)

def print_agent_report(svc: str, state: dict, llm_mode: str) -> None:
    rec = state.get("recommendation", {})
    bot = state.get("bottleneck", {})
    sd  = state.get("scaling_decision", {})
    mon_summary = state.get("monitoring_summary")
    m   = state.get("metrics", {})

    console.print()
    console.rule(f"[bold cyan]🤖 Agent Report — {svc.upper()}[/bold cyan]")

    # Agent 1 – Monitoring
    if mon_summary:
        mon_text = mon_summary
    else:
        mon_text = (
            f"CPU [bold]{m.get('cpu_percent',0):.1f}%[/bold]  "
            f"Memory [bold]{m.get('memory_percent',0):.1f}%[/bold] ({m.get('memory_mb',0):.0f} MB)  "
            f"Req/s [bold]{m.get('request_rate',0):.1f}[/bold]  "
            f"P95 latency [bold]{m.get('p95_latency_ms',0):.0f} ms[/bold]  "
            f"Replicas [bold]{m.get('replicas',0)}[/bold]"
        )
    console.print(Panel(mon_text, title=f"[bold]Agent 1 · Monitoring[/bold]  {llm_mode}",
                        border_style="blue", padding=(0, 1)))

    # Agent 2 – Bottleneck
    bot_text = Text()
    bot_text.append("Bottleneck: ", style="bold")
    bot_text.append(bot.get("bottleneck", "?"), style="bold magenta")
    bot_text.append("  Severity: ", style="bold")
    bot_text.append_text(_severity_style(bot.get("severity", "?")))
    evidence = bot.get("evidence", [])
    if evidence:
        bot_text.append("\n\nEvidence:\n", style="dim")
        for ev in evidence:
            bot_text.append(f"  • {ev}\n", style="italic")
    console.print(Panel(bot_text, title="[bold]Agent 2 · Bottleneck[/bold]",
                        border_style="magenta", padding=(0, 1)))

    # Agent 3 – Scaling
    sc_text = Text()
    sc_text.append("Action: ", style="bold")
    sc_text.append_text(_action_style(sd.get("action", "?")))
    sc_text.append(f"\n\nReplicas: ", style="bold")
    sc_text.append(f"{sd.get('current_replicas','?')} → {sd.get('recommended_replicas','?')}",
                   style="bold yellow")
    sc_text.append(f"\n\nReason: ", style="bold")
    sc_text.append(sd.get("reason", ""), style="italic")
    console.print(Panel(sc_text, title="[bold]Agent 3 · Scaling[/bold]",
                        border_style="yellow", padding=(0, 1)))

    # Agent 4 – Recommendation
    conf = rec.get("confidence", 0)
    rec_text = Text()
    rec_text.append(f"Confidence: [{_bar(conf)}] {conf*100:.0f}%\n\n", style="bold green")
    rec_text.append(rec.get("reason", ""), style="italic white")
    console.print(Panel(rec_text, title="[bold]Agent 4 · Recommendation[/bold]",
                        border_style="green", padding=(0, 1)))

# ── Scenario runner ──────────────────────────────────────────────────────────

SCENARIO_LABELS = {
    "normal":        "🟢 Normal Load",
    "cpu_stress":    "🔴 CPU Stress",
    "memory_stress": "🟡 Memory Stress",
}

def run_scenario(scenario: str) -> None:
    label = SCENARIO_LABELS.get(scenario, scenario)
    console.print()
    console.print(Panel(
        f"[bold white]Scenario: {label}[/bold white]\n"
        "[dim]Spawning 3 micro-service processes · measuring real CPU/memory via psutil[/dim]",
        border_style="white", padding=(1, 2),
    ))

    procs: dict[str, subprocess.Popen] = {}
    try:
        spawn_services(scenario, procs)
        console.print(f"[dim]⏳ Warming up for {WARMUP_SECS}s…[/dim]")
        time.sleep(WARMUP_SECS)
        all_metrics = collect_all_metrics(procs)
    finally:
        # Always reap the load generators, even on Ctrl-C or a crash
        kill_services(procs)

    if not all_metrics:
        console.print("[red]No metrics collected. Aborting.[/red]")
        return

    print_metrics_table(all_metrics)

    for svc, metrics in all_metrics.items():
        console.print(f"\n[dim]🧠 Running 4-agent LangGraph pipeline for [bold]{svc}[/bold]…[/dim]")
        before = llm_snapshot()
        try:
            state = run_pipeline(svc, metrics)
        except Exception as e:
            console.print(f"[red]Pipeline failed for {svc}: {e}[/red]")
            continue

        if state.get("error"):
            console.print(f"[red]Pipeline error for {svc}: {state['error']}[/red]")
            continue

        # Report what actually produced this output, not what we hoped for
        llm_mode = f"[dim italic]({escape(llm_mode_label(before))})[/dim italic]"
        print_agent_report(svc, state, llm_mode)

    console.print()
    console.rule("[dim]End of scenario[/dim]")

# ── Interactive menu ─────────────────────────────────────────────────────────

def interactive_menu() -> None:
    console.print(Panel(
        "[bold cyan]🚀 Multi-Agent Container Resource Advisor[/bold cyan]\n"
        "[dim]LangGraph pipeline · 4 Gemini-powered agents · Real psutil metrics[/dim]",
        box=box.DOUBLE_EDGE, border_style="cyan", padding=(1, 4),
    ))
    options = list(SCENARIO_LABELS.items())
    for i, (_, label) in enumerate(options, 1):
        console.print(f"  [bold]{i}[/bold]. {label}")
    console.print(f"  [bold]4[/bold]. Run ALL scenarios")
    console.print(f"  [bold]q[/bold]. Quit\n")

    choice = console.input("[bold cyan]Select: [/bold cyan]").strip().lower()
    if choice == "q":
        return
    elif choice == "4":
        for key, _ in options:
            run_scenario(key)
    elif choice in ("1", "2", "3"):
        run_scenario(options[int(choice) - 1][0])
    else:
        console.print("[red]Invalid choice.[/red]")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Container Resource Advisor")
    parser.add_argument("--scenario", choices=["normal", "cpu_stress", "memory_stress"])
    parser.add_argument("--all", action="store_true", help="Run all 3 scenarios")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Gemini and run the rule-based agents only")
    args = parser.parse_args()

    if args.no_llm:
        disable_llm()
    elif not os.getenv("GEMINI_API_KEY"):
        console.print("[yellow]GEMINI_API_KEY not set (.env or environment) — "
                      "running rule-based agents only.[/yellow]")

    if args.all:
        for sc in ["normal", "cpu_stress", "memory_stress"]:
            run_scenario(sc)
    elif args.scenario:
        run_scenario(args.scenario)
    else:
        interactive_menu()
