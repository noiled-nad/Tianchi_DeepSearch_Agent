"""节点：finalize

子任务汇总版：
1) 汇总所有子任务的 findings，结合原始文档生成答案
2) 判断是否需要继续检索
3) 如需继续，产出 follow-up queries
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List

from langchain_core.messages import AIMessage

from ..prompt_loader import load_prompt
from ..schemas import Document
from ..state import DeepResearchState
from ..memory import ExecutionMemory


FINALIZE_PROMPT = load_prompt("finalize.yaml", "finalize_prompt")


def _preview_block(text: str, max_chars: int = 1200) -> str:
    """压缩调试输出，避免日志过长。"""
    normalized = (text or "").strip()
    if not normalized:
        return "（空）"
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "\n...[截断]"


def _format_sources_full(docs: List[Document], max_chars_each: int = 1200) -> str:
    """完整证据包（仅在无 subtask_findings 时使用）。"""
    if not docs:
        return "（无可用证据文档）"
    chunks = []
    for i, d in enumerate(docs, start=1):
        title = (d.title or "").strip().replace("\n", " ")
        content = d.content or ""
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "\n[内容截断]"
        chunks.append(f"[S{i}] {title}\nURL: {d.url}\n内容:\n{content}\n")
    return "\n\n".join(chunks)


def _format_sources_index(docs: List[Document]) -> str:
    """轻量引用索引：仅标题+URL，用于子任务 findings 已有时。"""
    if not docs:
        return ""
    lines = []
    for i, d in enumerate(docs, start=1):
        title = (d.title or "").strip().replace("\n", " ")
        lines.append(f"[S{i}] {title} | {d.url}")
    return "\n".join(lines)


def _safe_json_obj(text: str) -> Dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    t = t.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def _format_subtask_findings(subtask_findings: Dict[str, any], subtasks: List[Dict]) -> str:
    """将子任务结构化发现格式化为文本，支持 (sub_q, evidence, candidates) 结构。"""
    if not subtask_findings:
        return ""
    lines = ["## 子任务调查结果"]
    st_map = {st["id"]: st for st in subtasks} if subtasks else {}
    for st_id, findings in subtask_findings.items():
        st = st_map.get(st_id, {})
        title = st.get("title", st_id)
        lines.append(f"\n### [{st_id}] {title}")
        if isinstance(findings, dict):
            # 结构化输出：(sub_q, evidence, candidates)
            sub_q = findings.get("sub_query", "")
            candidates = findings.get("candidates", [])
            evidence = findings.get("evidence", [])
            confidence = findings.get("confidence", 0)
            sources = findings.get("sources", [])
            lines.append(f"**核心问题**: {sub_q}")
            if len(candidates) > 1:
                lines.append(f"**最佳答案**: {candidates[0]} (置信度: {confidence})")
                lines.append(f"**候选答案**: {', '.join(candidates)}")
            elif candidates:
                lines.append(f"**答案**: {candidates[0]} (置信度: {confidence})")
            else:
                lines.append(f"**答案**: (未找到) (置信度: {confidence})")
            if evidence:
                lines.append("**证据**:")
                for e in evidence:
                    lines.append(f"  - {e}")
            if sources:
                lines.append("**来源**: " + ", ".join(sources[:3]))
        else:
            # 兼容旧格式（纯字符串）
            lines.append(str(findings))
    return "\n".join(lines)


def make_finalize_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def finalize(state: DeepResearchState) -> DeepResearchState:
        print("\n============ finalize 阶段 ============")
        question: str = state.get("question", "")
        brief = state.get("research_brief", {}) or {}
        docs: List[Document] = state.get("documents", []) or []
        subtask_findings = state.get("subtask_findings", {}) or {}
        subtasks = state.get("subtasks", []) or []
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 4))
        gaps = state.get("research_gaps", []) or []

        print(f"[finalize] iteration={iteration}/{max_iterations}, "
              f"documents={len(docs)}, subtask_findings={len(subtask_findings)}, gaps={len(gaps)}")

        # 构建证据：子任务发现 + 原始文档
        findings_text = _format_subtask_findings(subtask_findings, subtasks)

        # ── 执行记忆：推理链上下文 ──
        memory_context = ""
        raw_mem = state.get("execution_memory") or {}
        if raw_mem:
            memory = ExecutionMemory.from_dict(raw_mem)
            memory_context = memory.to_context_for_finalize()
            if memory_context:
                findings_text = memory_context + "\n\n" + findings_text
                print(f"[finalize] 已注入执行记忆上下文 ({len(memory_context)} chars)")

        # 有 subtask_findings 时只附轻量索引，否则灌全文
        if subtask_findings:
            sources_text = _format_sources_index(docs)
            sources_label = "文档引用索引"
        else:
            sources_text = _format_sources_full(docs)
            sources_label = "原始证据包"

        print(f"[finalize] findings_text_len={len(findings_text)}, sources_text_len={len(sources_text)}")
        print(f"[finalize] findings_text_preview:\n{_preview_block(findings_text)}")
        print(f"[finalize] {sources_label}_preview:\n{_preview_block(sources_text)}")
        if gaps:
            print(f"[finalize] known_gaps: {gaps}")

        prompt = FINALIZE_PROMPT.format(
            question=question,
            gaps_json=json.dumps(gaps, ensure_ascii=False),
            findings_text=findings_text,
            sources_label=sources_label,
            sources_text=sources_text,
        )

        print(f"[finalize] prompt_len={len(prompt)} chars")
        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        print(f"[finalize] raw_len={len(raw)}")

        try:
            obj = _safe_json_obj(raw)
        except Exception:
            obj = {}

        reasoning = str(obj.get("reasoning", "")).strip()
        if reasoning:
            print(f"[finalize] reasoning: {reasoning[:200]}...")

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
