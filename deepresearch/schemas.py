from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """搜索得到的结果"""
    title: str
    url: str
    snippet: Optional[str] = None


class Document(BaseModel):
    """抓取网页或 PDF得到的文档内容"""
    url: str
    title: Optional[str] = None
    content: str = Field(..., description="正文文本")


@dataclass
class SubtaskResult:
    """
    子任务执行结果（结构化）。

    用于替代原来的纯文本 findings，记录完整的执行过程和结果。
    """
    # 基本信息
    subtask_id: str
    title: str

    # 执行过程
    queries_original: List[str] = field(default_factory=list)      # 原始查询
    queries_reflected: List[str] = field(default_factory=list)     # reflect 后的
    queries_rollout: List[str] = field(default_factory=list)       # rollout 后的
    queries_final: List[str] = field(default_factory=list)         # 最终执行的查询
    search_results_count: int = 0
    docs_fetched_urls: List[str] = field(default_factory=list)     # 抓取的文档 URLs
    docs_fetched_count: int = 0

    # 结果
    findings: str = ""                                             # LLM 抽取的关键发现
    entities_found: List[str] = field(default_factory=list)        # 新发现的实体

    # 元数据
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0
    llm_calls: int = 0                                             # LLM 调用次数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化和状态存储）"""
        return {
            "subtask_id": self.subtask_id,
            "title": self.title,
            "queries_original": self.queries_original,
            "queries_reflected": self.queries_reflected,
            "queries_rollout": self.queries_rollout,
            "queries_final": self.queries_final,
            "search_results_count": self.search_results_count,
            "docs_fetched_urls": self.docs_fetched_urls,
            "docs_fetched_count": self.docs_fetched_count,
            "findings": self.findings,
            "entities_found": self.entities_found,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskResult":
        """从字典恢复"""
        return cls(
            subtask_id=data.get("subtask_id", ""),
            title=data.get("title", ""),
            queries_original=data.get("queries_original", []),
            queries_reflected=data.get("queries_reflected", []),
            queries_rollout=data.get("queries_rollout", []),
            queries_final=data.get("queries_final", []),
            search_results_count=data.get("search_results_count", 0),
            docs_fetched_urls=data.get("docs_fetched_urls", []),
            docs_fetched_count=data.get("docs_fetched_count", 0),
            findings=data.get("findings", ""),
            entities_found=data.get("entities_found", []),
            success=data.get("success", True),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0.0),
            llm_calls=data.get("llm_calls", 0),
        )

    def get_findings_text(self) -> str:
        """获取用于 prompt 的 findings 文本（兼容旧代码）"""
        if not self.findings:
            return "未找到相关信息。"
        return self.findings
