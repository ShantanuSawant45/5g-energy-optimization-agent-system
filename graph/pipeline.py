"""
graph/pipeline.py
Wires all 4 agents into a LangGraph StateGraph pipeline.
"""

from langgraph.graph import StateGraph, END
from graph.state import NetworkState
from agents.monitor_agent      import monitor_agent
from agents.prediction_agent   import prediction_agent
from agents.optimization_agent import optimization_agent
from agents.control_agent      import control_agent


def build_pipeline() -> StateGraph:
    graph = StateGraph(NetworkState)

    # Register nodes
    graph.add_node("monitor",  monitor_agent)
    graph.add_node("predict",  prediction_agent)
    graph.add_node("optimize", optimization_agent)
    graph.add_node("control",  control_agent)

    # Wire edges
    graph.add_edge("monitor",  "predict")
    graph.add_edge("predict",  "optimize")
    graph.add_edge("optimize", "control")
    graph.add_edge("control",  END)

    graph.set_entry_point("monitor")

    return graph.compile()