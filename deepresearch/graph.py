# deepresearch/graph.py
# -*- coding: utf-8 -*-
"""
新版流程（更少节点，弱化 SPOQ）：
START -> parse_claims(brief) -> retrieve -> finalize -> (retrieve|END)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import END, START, StateGraph

from deepresearch.state import DeepResearchState
from deepresearch.nodes.parse_claims import make_parse_claims_node
from deepresearch.nodes.retrieve import make_retrieve_node
from deepresearch.nodes.finalize import make_finalize_node


def _route_after_finalize(state: DeepResearchState) -> str:
    it = int(state.get("iteration", 0))
    max_it = int(state.get("max_iterations", 4))
    needs_followup = bool(state.get("needs_followup", False))

    if needs_followup and it < max_it:
        return "retrieve"
    return "end"

def build_deepresearch_graph(llm, searcher, fetcher):
    g = StateGraph(DeepResearchState)

    g.add_node("parse_claims", make_parse_claims_node(llm))
    g.add_node("retrieve", make_retrieve_node(searcher, fetcher))
    g.add_node("finalize", make_finalize_node(llm))

    g.add_edge(START, "parse_claims")
    g.add_edge("parse_claims", "retrieve")
    g.add_edge("retrieve", "finalize")

    g.add_conditional_edges("finalize", _route_after_finalize, {
        "retrieve": "retrieve",
        "end": END,
    })
    return g

if __name__ == "__main__":

    # Dummy components for graph visualization
    class _DummyLLM:
        pass

    class _DummySearcher:
        pass

    class _DummyFetcher:
        pass

    # Build the graph
    llm = _DummyLLM()
    searcher = _DummySearcher()
    fetcher = _DummyFetcher()
    graph = build_deepresearch_graph(llm, searcher, fetcher)

    # Compile the graph
    compiled_graph = graph.compile()

    # Draw the graph as Mermaid diagram
    mermaid_code = compiled_graph.get_graph().draw_mermaid()
    print("=== DeepResearch Graph Mermaid Diagram ===")
    print(mermaid_code)

    # Optionally save to file
    with open("deepresearch_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("Mermaid diagram saved to deepresearch_graph.mmd")