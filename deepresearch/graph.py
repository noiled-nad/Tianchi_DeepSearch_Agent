# deepresearch/graph.py
# -*- coding: utf-8 -*-
"""
OAgents 风格子任务拆解流程：
START -> parse_claims(subtask planning) -> execute_subtasks -> finalize -> (execute_subtasks|END)

execute_subtasks 内部按 parallel_groups 分层并行：
  每个子任务: query_optimize → search → fetch → extract_findings
  后序子任务注入前序 findings（级联推理）
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import END, START, StateGraph

from deepresearch.state import DeepResearchState
from deepresearch.nodes.parse_claims import make_parse_claims_node
from deepresearch.nodes.review_plan import make_review_plan_node
from deepresearch.nodes.execute_subtasks import make_execute_subtasks_node
from deepresearch.nodes.finalize import make_finalize_node
from deepresearch.nodes.replan import make_replan_node


def _route_after_replan(state: DeepResearchState) -> str:
    it = int(state.get("iteration", 0))
    max_it = int(state.get("max_iterations", 4))
    needs_followup = bool(state.get("needs_followup", False))

    if needs_followup and it < max_it:
        return "execute_subtasks"   # followup 走 execute_subtasks
    return "end"


def build_deepresearch_graph(llm, searcher, fetcher, flash_llm=None):
    g = StateGraph(DeepResearchState)

    g.add_node("parse_claims", make_parse_claims_node(llm))
    g.add_node("review_plan", make_review_plan_node(flash_llm or llm))
    g.add_node("execute_subtasks", make_execute_subtasks_node(llm, flash_llm, searcher, fetcher))
    g.add_node("finalize", make_finalize_node(flash_llm)) ## flash总结反思
    g.add_node("replan", make_replan_node())

    # START -> parse_claims -> review_plan -> execute_subtasks -> finalize -> replan
    g.add_edge(START, "parse_claims")
    g.add_edge("parse_claims", "review_plan")
    g.add_edge("review_plan", "execute_subtasks")
    g.add_edge("execute_subtasks", "finalize")
    g.add_edge("finalize", "replan")

    # replan -> execute_subtasks (循环) | END
    g.add_conditional_edges("replan", _route_after_replan, {
        "execute_subtasks": "execute_subtasks",
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
