"""Agent 4 – Recommendation Agent"""
from __future__ import annotations
import logging
from agents.state import (AgentState, FinalRecommendation,
                          MetricsData, BottleneckData, ScalingDecision)
from agents.llm_factory import call_gemini, QuotaExhausted

logger = logging.getLogger(__name__)


def _confidence(m: MetricsData, b: BottleneckData) -> float:
    score = 0.5
    if b.severity == "HIGH":    score += 0.30
    elif b.severity == "MEDIUM":score += 0.20
    elif b.severity == "LOW":   score += 0.10
    if b.bottleneck != "NONE":  score += 0.10
    if m.cpu_percent > 80 or m.memory_percent > 80:
        score += 0.10
    return round(min(score, 1.0), 2)


def _rule_narrative(m: MetricsData, b: BottleneckData, sd: ScalingDecision) -> str:
    """Generate a deterministic human-readable narrative when Gemini is unavailable."""
    parts = []
    if b.bottleneck == "CPU" and b.severity == "HIGH":
        parts.append(
            f"The {m.service} service is critically CPU-saturated at {m.cpu_percent:.0f}%, "
            f"causing P95 latency to spike to {m.p95_latency_ms:.0f} ms."
        )
        parts.append(
            f"Scaling from {sd.current_replicas} to {sd.recommended_replicas} replicas "
            f"will distribute load and bring CPU utilisation below the 75% safety threshold."
        )
        parts.append(
            "After scaling, monitor CPU per-replica and P95 latency; "
            "if latency remains elevated, consider memory limits or application profiling."
        )
    elif b.bottleneck == "MEMORY" and b.severity in ("HIGH", "MEDIUM"):
        parts.append(
            f"Memory usage on {m.service} is at {m.memory_percent:.0f}% ({m.memory_mb:.0f} MB), "
            f"approaching exhaustion."
        )
        parts.append(
            f"Adding replicas ({sd.current_replicas} → {sd.recommended_replicas}) reduces "
            "per-replica memory pressure and prevents OOM kills."
        )
        parts.append(
            "After scaling, watch for memory leaks; if usage continues climbing, "
            "review application heap settings."
        )
    elif sd.action == "NO_ACTION":
        parts.append(
            f"The {m.service} service is healthy: CPU {m.cpu_percent:.0f}%, "
            f"memory {m.memory_percent:.0f}%, latency {m.p95_latency_ms:.0f} ms."
        )
        parts.append("No scaling action is required at this time.")
        parts.append("Continue monitoring; re-analyse if traffic spikes.")
    else:
        parts.append(sd.reason)
    return " ".join(parts)


def recommendation_agent_node(state: AgentState) -> AgentState:
    raw_m = state.get("metrics"); raw_b = state.get("bottleneck")
    raw_sd = state.get("scaling_decision")
    if not all([raw_m, raw_b, raw_sd]):
        return {**state, "error": "Missing data for RecommendationAgent"}

    m  = MetricsData(**raw_m)
    b  = BottleneckData(**raw_b)
    sd = ScalingDecision(**raw_sd)
    conf = _confidence(m, b)

    prompt = f"""You are a senior SRE writing a scaling recommendation.
Service: {m.service}
CPU {m.cpu_percent:.1f}% | Memory {m.memory_percent:.1f}% ({m.memory_mb:.0f} MB) | {m.replicas} replicas
Req/s {m.request_rate:.1f} | P95 latency {m.p95_latency_ms:.0f} ms
Bottleneck: {b.bottleneck} ({b.severity})  Evidence: {b.evidence}
Decision: {sd.action}  Replicas: {sd.current_replicas} → {sd.recommended_replicas}
Confidence: {conf*100:.0f}%
Write exactly 3 clear sentences: (1) what the problem is, (2) why scaling helps, (3) what to watch after."""

    try:
        narrative = call_gemini(prompt)
        logger.info("[RecommendationAgent] LLM narrative generated")
    except QuotaExhausted:
        narrative = _rule_narrative(m, b, sd)
        logger.info("[RecommendationAgent] Using rule-based narrative")
    except Exception as e:
        logger.debug("RecommendationAgent LLM error: %s", e)
        narrative = _rule_narrative(m, b, sd)

    result = FinalRecommendation(
        service=m.service, action=sd.action,  # type: ignore
        current_replicas=sd.current_replicas,
        recommended_replicas=sd.recommended_replicas,
        bottleneck=b.bottleneck, severity=b.severity,
        confidence=conf, reason=narrative, evidence=b.evidence,
    )
    logger.info("[RecommendationAgent] Done. action=%s conf=%.2f", result.action, result.confidence)
    return {**state, "recommendation": result.model_dump()}
