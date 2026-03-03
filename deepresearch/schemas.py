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
