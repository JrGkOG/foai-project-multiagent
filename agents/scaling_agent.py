"""Agent 3 – Scaling Agent"""
from __future__ import annotations
import json, re, logging
from agents.state import AgentState, ScalingDecision, MetricsData, BottleneckData
from agents.llm_factory import call_gemini, QuotaExhausted

logger = logging.getLogger(__name__)
MIN_R, MAX_R = 1, 10


def _deterministic(m: MetricsData, b: BottleneckData) -> ScalingDecision:
    cur = m.replicas
    if b.severity == "HIGH":
        return ScalingDecision(action="SCALE_UP", current_replicas=cur,
                               recommended_replicas=min(cur * 2, MAX_R),
                               reason=f"{b.bottleneck} bottleneck at HIGH severity — need more capacity.")
    if b.severity == "MEDIUM":
        return ScalingDecision(action="SCALE_UP", current_replicas=cur,
                               recommended_replicas=min(cur + 1, MAX_R),
                               reason=f"{b.bottleneck} bottleneck at MEDIUM severity — adding 1 replica.")
    if cur > 2 and b.severity in ("NONE", "LOW"):
        return ScalingDecision(action="SCALE_DOWN", current_replicas=cur,
                               recommended_replicas=max(MIN_R, cur - 1),
                               reason="Resources healthy — scaling down to save capacity.")
    return ScalingDecision(action="NO_ACTION", current_replicas=cur,
                           recommended_replicas=cur,
                           reason="Metrics within acceptable range — no action needed.")


def _parse(text: str, m: MetricsData, b: BottleneckData) -> ScalingDecision:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            action = str(d.get("action", "NO_ACTION")).upper()
            if action not in {"SCALE_UP", "SCALE_DOWN", "NO_ACTION"}:
                action = "NO_ACTION"
            rec = max(MIN_R, min(MAX_R, int(d.get("recommended_replicas", m.replicas))))
            return ScalingDecision(action=action, current_replicas=m.replicas,  # type: ignore
                                   recommended_replicas=rec, reason=str(d.get("reason", "")))
        except Exception:
            pass
    return _deterministic(m, b)


def scaling_agent_node(state: AgentState) -> AgentState:
    raw_m = state.get("metrics"); raw_b = state.get("bottleneck")
    if not raw_m or not raw_b:
        return {**state, "error": "Missing data for ScalingAgent"}
    m = MetricsData(**raw_m); b = BottleneckData(**raw_b)

    prompt = f"""You are a Kubernetes autoscaling engine.
Service: {m.service}  Current replicas: {m.replicas}
CPU: {m.cpu_percent:.1f}%  Memory: {m.memory_percent:.1f}%  P95 lat: {m.p95_latency_ms:.0f} ms
Bottleneck: {b.bottleneck} ({b.severity})
Replicas MUST be {MIN_R}–{MAX_R}. Reply ONLY with valid JSON (no markdown):
{{"action":"SCALE_UP|SCALE_DOWN|NO_ACTION","recommended_replicas":<int>,"reason":"<reason>"}}"""

    try:
        result = _parse(call_gemini(prompt), m, b)
    except QuotaExhausted:
        result = _deterministic(m, b)
    except Exception as e:
        logger.debug("ScalingAgent LLM error: %s", e)
        result = _deterministic(m, b)

    logger.info("[ScalingAgent] %s  %d→%d", result.action,
                result.current_replicas, result.recommended_replicas)
    return {**state, "scaling_decision": result.model_dump()}
