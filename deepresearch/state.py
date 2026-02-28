# deepresearch/state.py
# -*- coding: utf-8 -*-
"""
LangGraph State
"""
from __future__ import annotations
from typing import Annotated, List, TypedDict, Dict, Any, Literal
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .schemas import Claim, Document, Verdict, Evidence


class DeepResearchState(TypedDict, total=False):
    # 对话上下文
    messages: Annotated[List[BaseMessage], add_messages]

    # 问题
    question: str
    # 旧版结构化约束（兼容保留，不再作为主流程依赖）
    claims: List[Claim]

    # 新版：研究简报（不再强依赖 SPOQ）
    research_brief: Dict[str, Any]

    # 搜索内容
    queries: List[str]
    query_history: List[str]
    research_gaps: List[str]
    needs_followup: bool

    # 得到的结果
    documents: List[Document]

    # ✅ 长期记忆：证据片段（兼容保留）
    notes: Annotated[List[Evidence], operator.add]

    # ✅ 判定结果（兼容保留）
    claim_verdicts: Dict[str, Verdict]
    missing_info: List[str]


    final_answer: str
    # 调度参数
    difficulty: Literal["low", "mid", "high"]         # "low" | "mid" | "high"
    batch_size: int          # e.g. 3~8
    parallelism: int         # e.g. 1~5

    # ====== (Optional for next step loop) ======
    iteration: int
    max_iterations: int
