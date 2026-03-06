# deepresearch/state.py
# -*- coding: utf-8 -*-
"""
LangGraph State
"""
from __future__ import annotations
from typing import Annotated, List, TypedDict, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .schemas import Document


class DeepResearchState(TypedDict, total=False):
    # 对话上下文
    messages: Annotated[List[BaseMessage], add_messages]

    # 问题
    question: str

    # 研究简报
    research_brief: Dict[str, Any]

    # ── 子任务拆解（OAgents 风格） ──
    # 每个 subtask: {id, title, queries, depends_on, reason}
    subtasks: List[Dict[str, Any]]
    # 按依赖层级分组：[["ST1","ST2"], ["ST3"]]，同组可并行
    parallel_groups: List[List[str]]
    # 每个子任务的结构化发现：{"ST1": {sub_query, evidence, candidates, confidence, sources}, ...}
    subtask_findings: Dict[str, Any]

    # 搜索内容（保留兼容）
    queries: List[str]
    query_history: List[str]
    research_gaps: List[str]
    needs_followup: bool

    # 得到的结果
    documents: List[Document]

    final_answer: str
    # 循环控制
    iteration: int
    max_iterations: int
