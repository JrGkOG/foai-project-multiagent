"""Agent 1 – Monitoring Agent"""
from __future__ import annotations
import logging
from agents.state import AgentState, MetricsData
from agents.llm_factory import call_gemini, LLMUnavailable

logger = logging.getLogger(__name__)


def monitoring_agent_node(state: AgentState) -> AgentState:
    raw = state.get("metrics")
    if not raw:
        return {**state, "error": "No metrics in state"}
    try:
        m = MetricsData(**raw)
    except Exception as e:
        return {**state, "error": f"MetricsData validation failed: {e}"}

    prompt = f"""You are a Kubernetes monitoring agent.
Real-time metrics for '{m.service}':
CPU {m.cpu_percent:.1f}% | Memory {m.memory_percent:.1f}% ({m.memory_mb:.0f} MB) | {m.replicas} replicas
Req/s {m.request_rate:.1f} | P95 latency {m.p95_latency_ms:.0f} ms
In 2 sentences describe the current health. Be factual and concise."""

    try:
        summary = call_gemini(prompt)
        logger.info("[MonitoringAgent] LLM summary: %s", summary)
    except LLMUnavailable:
        summary = None   # will use rule-based display
    except Exception as e:
        summary = None
        logger.debug("MonitoringAgent LLM error: %s", e)

    return {**state, "metrics": m.model_dump(), "monitoring_summary": summary}
