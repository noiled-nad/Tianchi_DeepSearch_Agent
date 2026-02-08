# deepresearch/state.py
# -*- coding: utf-8 -*-
"""
LangGraph State
"""
from __future__ import annotations
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages   

from .schemas import Claim, Document

class DeepResearchState(TypedDict, total=False):## total=False
    ## 对话上下文，基于annotated方法，给list添加add reducer
    messages: Annotated[List[BaseMessage],add_messages]

    ## 问题
    question :str
    # 解析得到的约束列表
    claims: List[Claim]
    ## 搜索内容
    queries:List[str]
    ## 得到的结果
    documents:List[Document]

    final_answer:str