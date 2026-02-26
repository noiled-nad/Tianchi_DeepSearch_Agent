"""节点：finalize

新版职责：
1) 基于现有证据生成答案草稿
2) 判断是否需要继续检索
3) 如需继续，直接产出下一轮 follow-up queries
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List

from langchain_core.messages import AIMessage

from ..schemas import Document
from ..state import DeepResearchState


def _format_sources(docs: List[Document], max_chars_each: int = 1600) -> str:
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


def _safe_json_obj(text: str) -> Dict:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def make_finalize_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def finalize(state: DeepResearchState) -> DeepResearchState:
        print("\n============ finalize 阶段 ============")
        question: str = state.get("question", "")
        brief = state.get("research_brief", {}) or {}
        docs: List[Document] = state.get("documents", []) or []
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 4))
        gaps = state.get("research_gaps", []) or []
        print(f"[finalize] iteration={iteration}/{max_iterations}, documents={len(docs)}, gaps={len(gaps)}")

        sources_text = _format_sources(docs)
        prompt = (
            "你是 Deep Research 的综合与决策器。\n"
            "请基于证据包输出一个 JSON 对象（不要 markdown）：\n"
            "{\n"
            "  \"final_answer\": string,\n"
            "  \"confidence\": number,\n"
            "  \"needs_followup\": boolean,\n"
            "  \"research_gaps\": [string],\n"
            "  \"followup_queries\": [string]\n"
            "}\n"
            "规则：\n"
            "1) final_answer 必须可直接交付；关键句用 [S1]/[S2] 引用。\n"
            "2) 若证据不足，needs_followup=true，并给 3~6 条 followup_queries。\n"
            "3) followup_queries 必须具体，可直接搜索。\n"
            "4) 若证据已充分，needs_followup=false，followup_queries 置空。\n\n"
            f"问题：{question}\n"
            f"研究简报：{json.dumps(brief, ensure_ascii=False)}\n"
            f"已知缺口：{json.dumps(gaps, ensure_ascii=False)}\n"
            f"证据包：\n{sources_text}\n"
        )

        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        print(f"[finalize] raw_len={len(raw)}")

        try:
            obj = _safe_json_obj(raw)
        except Exception:
            obj = {}

        answer_text = str(obj.get("final_answer", "")).strip()
        if not answer_text:
            answer_text = "Final Answer: Unknown\n\nEvidence: 当前证据不足，无法稳定锁定唯一答案。"

        next_queries = [str(x).strip() for x in obj.get("followup_queries", []) if str(x).strip()]
        if len(next_queries) > 6:
            next_queries = next_queries[:6]

        model_need_followup = bool(obj.get("needs_followup", False))
        has_room = (iteration + 1) < max_iterations
        need_followup = model_need_followup and has_room and bool(next_queries)

        new_gaps = [str(x).strip() for x in obj.get("research_gaps", []) if str(x).strip()]
        progress = AIMessage(
            content=(
                f"[finalize] 置信度={obj.get('confidence', 0)}，"
                f"needs_followup={need_followup}，下一轮查询={len(next_queries) if need_followup else 0}。"
            )
        )

        return {
            "final_answer": answer_text,
            "needs_followup": need_followup,
            "research_gaps": new_gaps,
            "queries": next_queries if need_followup else [],
            "iteration": iteration + 1,
            "messages": [progress, AIMessage(content=answer_text)],
        }

    return finalize
