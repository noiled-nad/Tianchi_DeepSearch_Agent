# deepresearch/nodes/finalize.py
# -*- coding: utf-8 -*-
"""
节点4：finalize
目标：基于 documents 生成最终回答，并要求“文内引用 [S1][S2]...”。

- 不做复杂验证/候选池
- 输出：答案 + 引用 + 来源列表
- 证据不足时也要输出：无法唯一确定 + 需要补的证据
"""

from __future__ import annotations

from typing import Callable, List

from langchain_core.messages import AIMessage

from ..schemas import Claim, Document
from ..state import DeepResearchState


def _format_sources(docs: List[Document], max_chars_each: int = 1800) -> str:
    """
    把抓到的文档整理成“证据包”，并限制每篇长度，防止上下文太长。
    """
    if not docs:
        return "（无可用证据文档）"

    chunks = []
    for i, d in enumerate(docs, start=1):
        title = (d.title or "").strip().replace("\n", " ")
        content = d.content or ""
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "\n[内容截断]"
        chunks.append(
            f"[S{i}] {title}\nURL: {d.url}\n内容:\n{content}\n"
        )
    return "\n\n".join(chunks)


def make_finalize_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def finalize(state: DeepResearchState) -> DeepResearchState:
        question: str = state.get("question", "")
        claims: List[Claim] = state.get("claims", [])
        docs: List[Document] = state.get("documents", [])

        claims_text = "\n".join([f"- {c.id}: {c.description}" for c in claims]) if claims else "（无）"
        sources_text = _format_sources(docs)

        prompt = (
            "你是一个deep research助手。你必须严格基于【证据包】回答问题。\n"
            "输出要求（必须遵守）：\n"
            "1) 第一行：Final Answer: <只写答案本体，不要多余解释>\n"
            "2) 第二部分：Evidence（分点列出），每个关键句末尾必须带引用，如 [S1] 或 [S2][S3]\n"
            "3) 最后：Sources 列表，格式：S1: url\n"
            "4) 若证据不足以唯一确定答案：Final Answer 仍要给你认为最可能的候选（或写 Unknown），并明确说明缺了什么证据。\n\n"
            f"问题：{question}\n\n"
            f"约束（供对齐）：\n{claims_text}\n\n"
            f"证据包：\n{sources_text}\n"
        )

        resp = await llm.ainvoke(prompt)
        answer_text = str(resp.content).strip()
        return {
            "final_answer": answer_text,
            "messages": [AIMessage(content=answer_text)],
        }

    return finalize
