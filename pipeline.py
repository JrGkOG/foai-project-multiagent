"""
LangGraph Pipeline
==================
Wires the four agents into a sequential graph:

  START → monitoring_node → bottleneck_node → scaling_node → recommendation_node → END

Any node that sets state["error"] short-circuits the graph straight to END.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.monitoring_agent     import monitoring_agent_node
from agents.bottleneck_agent     import bottleneck_agent_node
from agents.scaling_agent        import scaling_agent_node
from agents.recommendation_agent import recommendation_agent_node


def _next_or_end(next_node: str):
    """Router: continue to `next_node`, or stop if the previous node errored."""
    def router(state: AgentState) -> str:
        return END if state.get("error") else next_node
    return router


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Node names must not clash with AgentState field names
    g.add_node("monitoring_node",     monitoring_agent_node)
    g.add_node("bottleneck_node",     bottleneck_agent_node)
    g.add_node("scaling_node",        scaling_agent_node)
    g.add_node("recommendation_node", recommendation_agent_node)

    g.add_edge(START, "monitoring_node")
    g.add_conditional_edges("monitoring_node", _next_or_end("bottleneck_node"),
                            ["bottleneck_node", END])
    g.add_conditional_edges("bottleneck_node", _next_or_end("scaling_node"),
                            ["scaling_node", END])
    g.add_conditional_edges("scaling_node", _next_or_end("recommendation_node"),
                            ["recommendation_node", END])
    g.add_edge("recommendation_node", END)

    return g.compile()


# Singleton – reuse the compiled graph across calls
ADVISOR_GRAPH = build_graph()


def run_pipeline(service: str, metrics: dict) -> dict:
    """Execute the full agent pipeline and return the final state."""
    initial: AgentState = {
        "service": service,
        "metrics": metrics,
    }
    return ADVISOR_GRAPH.invoke(initial)
