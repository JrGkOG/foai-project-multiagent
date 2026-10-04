"""Agent 2 – Bottleneck Agent"""
from __future__ import annotations
import json, re, logging
from agents.state import AgentState, BottleneckData, MetricsData
from agents.llm_factory import call_gemini, LLMUnavailable

logger = logging.getLogger(__name__)

CPU_HIGH, CPU_MED = 75.0, 55.0
MEM_HIGH, MEM_MED = 80.0, 65.0
LAT_HIGH, LAT_MED = 350.0, 200.0


def _deterministic(m: MetricsData) -> BottleneckData:
    evidence, bottleneck, severity = [], "NONE", "NONE"
    if m.cpu_percent >= CPU_HIGH:
        bottleneck, severity = "CPU", "HIGH"
        evidence.append(f"CPU at {m.cpu_percent:.1f}% ≥ {CPU_HIGH}% threshold")
    elif m.cpu_percent >= CPU_MED:
        bottleneck, severity = "CPU", "MEDIUM"
        evidence.append(f"CPU at {m.cpu_percent:.1f}% ≥ {CPU_MED}% threshold")
    if m.memory_percent >= MEM_HIGH:
        if bottleneck == "NONE" or severity == "MEDIUM":
            bottleneck, severity = "MEMORY", "HIGH"
        evidence.append(f"Memory at {m.memory_percent:.1f}% ≥ {MEM_HIGH}%")
    elif m.memory_percent >= MEM_MED:
        if bottleneck == "NONE":
            bottleneck, severity = "MEMORY", "MEDIUM"
        evidence.append(f"Memory at {m.memory_percent:.1f}%")
    if m.p95_latency_ms >= LAT_HIGH:
        evidence.append(f"P95 latency {m.p95_latency_ms:.0f} ms critically high")
        if severity != "HIGH":
            bottleneck = bottleneck if bottleneck != "NONE" else "LATENCY"
            severity = "HIGH"
    elif m.p95_latency_ms >= LAT_MED:
        evidence.append(f"P95 latency {m.p95_latency_ms:.0f} ms elevated")
    if m.replicas <= 1 and bottleneck != "NONE":
        evidence.append(f"Only {m.replicas} replica(s)")
    return BottleneckData(bottleneck=bottleneck, severity=severity, evidence=evidence)  # type: ignore


def _parse(text: str, m: MetricsData) -> BottleneckData:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            b = str(d.get("bottleneck", "NONE")).upper()
            s = str(d.get("severity",   "NONE")).upper()
            e = d.get("evidence", [])
            if b not in {"CPU","MEMORY","LATENCY","REPLICAS","NONE"}: b = "NONE"
            if s not in {"LOW","MEDIUM","HIGH","NONE"}: s = "NONE"
            return BottleneckData(bottleneck=b, severity=s, evidence=e)  # type: ignore
        except Exception:
            pass
    return _deterministic(m)


def bottleneck_agent_node(state: AgentState) -> AgentState:
    raw = state.get("metrics")
    if not raw:
        return {**state, "error": "No metrics for BottleneckAgent"}
    m = MetricsData(**raw)

    prompt = f"""You are a Kubernetes performance analyst.
Service: {m.service}
CPU: {m.cpu_percent:.1f}%  Memory: {m.memory_percent:.1f}%  Replicas: {m.replicas}
Req/s: {m.request_rate:.1f}  P95 latency: {m.p95_latency_ms:.0f} ms
Reply ONLY with valid JSON (no markdown):
{{"bottleneck":"CPU|MEMORY|LATENCY|REPLICAS|NONE","severity":"LOW|MEDIUM|HIGH|NONE","evidence":["reason1","reason2"]}}"""

    try:
        result = _parse(call_gemini(prompt), m)
    except LLMUnavailable:
        result = _deterministic(m)
    except Exception as e:
        logger.debug("BottleneckAgent LLM error: %s", e)
        result = _deterministic(m)

    # Safety: trust deterministic rules on HIGH severity
    rule = _deterministic(m)
    if rule.severity == "HIGH" and result.severity != "HIGH":
        result = rule

    logger.info("[BottleneckAgent] %s / %s", result.bottleneck, result.severity)
    return {**state, "bottleneck": result.model_dump()}
