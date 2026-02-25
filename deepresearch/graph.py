# deepresearch/graph.py
# -*- coding: utf-8 -*-
"""
目前为线性流程：START -> parse_claims -> plan_queries -> retrieve -> finalize -> END
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import END, START, StateGraph

from deepresearch.state import DeepResearchState
from deepresearch.schemas import *
from deepresearch.nodes.parse_claims import make_parse_claims_node
from deepresearch.nodes.plan_queries import make_plan_queries_node
from deepresearch.nodes.retrieve import make_retrieve_node
from deepresearch.nodes.extract_notes import make_extract_notes_node
from deepresearch.nodes.judge_claims import make_judge_claims_node
from deepresearch.nodes.finalize import make_finalize_node


def _route_after_judge(state: DeepResearchState) -> str:
    """
    增加闭环
    """
    it = int(state.get("iteration", 0))
    max_it = int(state.get("max_iterations", 6))
    claims = state.get("claims", [])
    verdicts = state.get("claim_verdicts", {})

    # 终止：must 的关键类型都不是 unknown
    must_focus = [c for c in claims if c.must and c.claim_type in ("disamb", "anchor", "output")]
    still_unknown = [c for c in must_focus if verdicts.get(c.id, "unknown") == "unknown"]

    if not still_unknown:
        return "finalize"

    if it >= max_it:
        return "finalize"

    return "plan_queries"

def build_deepresearch_graph(llm, searcher, fetcher):
    g = StateGraph(DeepResearchState)

    g.add_node("parse_claims", make_parse_claims_node(llm))
    g.add_node("plan_queries", make_plan_queries_node(llm))
    g.add_node("retrieve", make_retrieve_node(searcher, fetcher))
    g.add_node("extract_notes", make_extract_notes_node(llm))
    g.add_node("judge_claims", make_judge_claims_node(llm))
    g.add_node("finalize", make_finalize_node(llm))

    g.add_edge(START, "parse_claims")
    g.add_edge("parse_claims", "plan_queries")
    g.add_edge("plan_queries", "retrieve")
    g.add_edge("retrieve", "extract_notes")
    g.add_edge("extract_notes", "judge_claims")

    g.add_conditional_edges("judge_claims", _route_after_judge, {
        "plan_queries": "plan_queries",
        "finalize": "finalize",
    })

    g.add_edge("finalize", END)
    return g

if __name__ == "__main__":
    from deepresearch.schemas import *  # Import all schemas to resolve type hints
    from deepresearch.nodes.parse_claims import make_parse_claims_node
    from deepresearch.nodes.route_and_schedule import make_route_and_schedule_node
    from deepresearch.nodes.plan_queries import make_plan_queries_node
    from deepresearch.nodes.retrieve import make_retrieve_node
    from deepresearch.nodes.finalize import make_finalize_node

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