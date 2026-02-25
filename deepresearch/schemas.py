from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


ClaimType = Literal["anchor", "disamb", "bridge", "output", "other"]

class EntityRef(BaseModel):
    """主体/客体可链接实体的最小表示"""
    name: str = Field(..., description="实体名称（原文或规范名）")
    type: Optional[str] = Field(default=None, description="实体类型（person/org/place/work/event/doc/...）")
    aliases: List[str] = Field(default_factory=list, description="别名/同名/译名等（可选）")

class Qualifier(BaseModel):
    time: Optional[str] = None
    location: Optional[str] = None
    quantity: Optional[Union[int, float, str]] = None
    disambiguation: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra", mode="before")
    @classmethod
    def normalize_extra(cls, v):
        # null -> {}
        if v is None:
            return {}
        # "人民币" -> {"currency": "人民币"} (或 "CNY")
        if isinstance(v, str):
            return {"note": v} if v.strip() else {}
        # dict 原样
        if isinstance(v, dict):
            return v
        # 其他类型也别炸，兜底转字符串
        return {"note": str(v)}

class Claim(BaseModel):
    id: str = Field(..., description="claim 唯一 id，比如 c1/c2")
    must: bool = Field(default=True, description="是否必须满足")

    # 分类
    claim_type: ClaimType = Field(default="other", description="anchor/disamb/bridge/output/other")

    # 四元组
    S: EntityRef = Field(..., description="Subject：主体实体")
    P: str = Field(..., description="Predicate：关系/谓词（尽量用简短英文动词短语）")
    O: Optional[EntityRef] = Field(default=None, description="Object：客体实体或属性（如年份）")

    # 限定条件
    Q: Qualifier = Field(default_factory=Qualifier, description="Qualifier：限定条件")

    # 便于调试/回退
    description: str = Field(..., description="一句话描述（中文），与四元组一致")
    
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

class FinalAnswer(BaseModel):
    """最终输出结构：答案 + 引用来源列表（S1..Sn）"""
    answer: str
    sources: List[str] = Field(default_factory=list, description="来源 url 列表，按顺序对应 S1/S2...")

Verdict = Literal["supported", "refuted", "unknown"]

class Evidence(BaseModel):
    """
    证据片段：绑定 claim_id，必须可引用（url + quote）。
    """
    claim_id: str
    url: str
    title: Optional[str] = None

    quote: str = Field(..., description="可引用的原文短摘录（建议 <= 300 字符）")
    snippet: Optional[str] = None

    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    stance: Optional[Literal["supports", "refutes", "neutral"]] = None
