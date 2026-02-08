# deepresearch/graph.py
# -*- coding: utf-8 -*-
"""
目前为线性流程：START -> parse_claims -> plan_queries -> retrieve -> finalize -> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .state import DeepResearchState
from .nodes.parse_claims import make_parse_claims_node
from .nodes.plan_queries import make_plan_queries_node
from .nodes.retrieve import make_retrieve_node
from .nodes.finalize import make_finalize_node

def build_deepresearch_graph(llm, searcher, fetcher):
    g = StateGraph(DeepResearchState)
    g.add_node("parse_claims", make_parse_claims_node(llm))
    g.add_node("plan_queries", make_plan_queries_node(llm))
    g.add_node("retrieve", make_retrieve_node(searcher, fetcher))
    g.add_node("finalize", make_finalize_node(llm))


    g.add_edge(START, "parse_claims")
    g.add_edge("parse_claims", "plan_queries")
    g.add_edge("plan_queries", "retrieve")
    g.add_edge("retrieve", "finalize")
    g.add_edge("finalize", END)
    return g