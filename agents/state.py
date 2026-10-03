"""
Shared AgentState + Pydantic models flowing through the LangGraph pipeline.
"""
from __future__ import annotations
from typing import Optional, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator


class MetricsData(BaseModel):
    service: str
    cpu_percent:    float = Field(..., ge=0, le=100)
    memory_percent: float = Field(..., ge=0, le=100)
    memory_mb:      float = 0.0
    replicas:       int   = Field(..., ge=0)
    request_rate:   float = Field(..., ge=0)
    p95_latency_ms: float = Field(..., ge=0)
    pid:            Optional[int] = None


class BottleneckData(BaseModel):
    bottleneck: Literal["CPU", "MEMORY", "LATENCY", "REPLICAS", "NONE"]
    severity:   Literal["LOW", "MEDIUM", "HIGH", "NONE"]
    evidence:   List[str] = Field(default_factory=list)


class ScalingDecision(BaseModel):
    action:               Literal["SCALE_UP", "SCALE_DOWN", "NO_ACTION"]
    current_replicas:     int
    # No ge/le constraint here – the validator below clamps BEFORE pydantic runs checks
    recommended_replicas: int
    reason:               str

    @field_validator("recommended_replicas", mode="before")
    @classmethod
    def clamp(cls, v) -> int:
        """Hard safety bound [1, 10]."""
        return max(1, min(10, int(v)))


class FinalRecommendation(BaseModel):
    service:              str
    action:               Literal["SCALE_UP", "SCALE_DOWN", "NO_ACTION"]
    current_replicas:     int
    recommended_replicas: int
    bottleneck:           str
    severity:             str
    confidence:           float = Field(..., ge=0, le=1)
    reason:               str
    evidence:             List[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Typed state dict flowing through every LangGraph node."""
    service:             str
    metrics:             Optional[dict]
    bottleneck:          Optional[dict]
    scaling_decision:    Optional[dict]
    recommendation:      Optional[dict]
    monitoring_summary:  Optional[str]
    error:               Optional[str]
