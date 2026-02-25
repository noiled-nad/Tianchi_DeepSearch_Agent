# deepresearch/state.py
# -*- coding: utf-8 -*-
"""
LangGraph State
"""
from __future__ import annotations
from typing import Annotated, List, TypedDict, Dict, Optional,Literal
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .schemas import Claim, Document, Verdict, Evidence


class DeepResearchState(TypedDict, total=False):
    # 对话上下文
    messages: Annotated[List[BaseMessage], add_messages]

    # 问题
    question: str
    # 解析得到的约束列表
    claims: List[Claim]

    # 搜索内容
    queries: List[str]
    # 得到的结果
    documents: List[Document]

    # ✅ 长期记忆：证据片段（不要再堆全文）
    notes: Annotated[List[Evidence], operator.add]

    # ✅ 判定结果
    claim_verdicts: Dict[str, Verdict]
    missing_info: List[str]  # 给下一轮 plan_queries 用的“缺口描述”


    final_answer: str
    # ====== NEW: Lemon-style scheduling signals ======
    difficulty: Literal["low", "mid", "high"]         # "low" | "mid" | "high"
    batch_size: int          # e.g. 3~8
    parallelism: int         # e.g. 1~5

    # ====== (Optional for next step loop) ======
    iteration: int
    max_iterations: int
