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

    # ── 约束提取 ──
    constraints: Dict[str, Any]

    # ── 子任务拆解（OAgents 风格） ──
    subtasks: List[Dict[str, Any]]
    parallel_groups: List[List[str]]
    subtask_findings: Dict[str, Any]

    # 搜索内容
    queries: List[str]
    query_history: List[str]
    research_gaps: List[str]
    needs_followup: bool

    # 结果
    documents: List[Document]
    final_answer: str

    # ── 推理链（与答案相关） ──
    reasoning_chain: List[Dict[str, Any]]

    # ── 验证结果 ──
    validation_result: str
    validation_passed: bool
    validation_confidence: float
    validation_reasoning: str
    missing_evidence: List[str]
    validation_suggestions: List[str]

    # 循环控制
    iteration: int
    max_iterations: int
