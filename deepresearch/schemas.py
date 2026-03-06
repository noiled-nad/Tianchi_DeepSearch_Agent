from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    ## 搜索得到的结果
    title: str
    url: str
    snippet: Optional[str] = None

class Document(BaseModel):
    ## 抓取网页或 PDF得到的文档内容
    url: str
    title: Optional[str] = None
    content: str = Field(..., description="正文文本")


class SubtaskResult(BaseModel):
    """子任务结构化输出: (sub_q, evidence, candidates)"""
    sub_query: str = Field("", description="子任务聚焦回答的核心问题")
    evidence: List[str] = Field(default_factory=list, description="支撑答案的关键证据列表")
    candidates: List[str] = Field(default_factory=list, description="所有符合条件的候选答案，第一个为最佳答案")
    confidence: float = Field(0.0, description="置信度 0.0~1.0")
    sources: List[str] = Field(default_factory=list, description="证据来源 URL")
