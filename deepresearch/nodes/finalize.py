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


FINALIZE_PROMPT = load_prompt("finalize.yaml", "finalize_prompt")


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
        # 有 subtask_findings 时只附轻量索引，否则灌全文
        if subtask_findings:
            sources_text = _format_sources_index(docs)
            sources_label = "文档引用索引"
        else:
            sources_text = _format_sources_full(docs)
            sources_label = "原始证据包"

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

        result = {
            "final_answer": answer_text,
            "reasoning_chain": reasoning_chain,
            "needs_followup": need_followup,
            "research_gaps": new_gaps,
            "queries": next_queries if need_followup else [],
            "iteration": iteration + 1,
            "messages": [progress, AIMessage(content=answer_text)],
        }

        # ── followup 策略：保留原子任务 + 为 gaps 新增定向子任务 ──
        if need_followup:
            # 复用原 subtask 结构
            updated_subtasks = list(subtasks) if subtasks else []
            updated_findings = dict(subtask_findings)
            existing_ids = {st["id"] for st in updated_subtasks}

            # 分析 LLM 指出的 research_gaps，生成针对性 gap 子任务
            for gi, gap in enumerate(new_gaps):
                gap_id = f"gap_{iteration}_{gi}"
                if gap_id not in existing_ids:
                    # 为每个 gap 分配相关的 followup_queries（均匀分配）
                    q_per_gap = max(1, len(next_queries) // max(1, len(new_gaps)))
                    start_idx = gi * q_per_gap
                    gap_queries = next_queries[start_idx:start_idx + q_per_gap]
                    if not gap_queries:
                        gap_queries = next_queries[:2] if next_queries else []

                    updated_subtasks.append({
                        "id": gap_id,
                        "title": f"补查: {gap[:60]}",
                        "reason": gap,
                        "queries": gap_queries,
                        "depends_on": [],  # gap 子任务可独立执行
                    })
                    existing_ids.add(gap_id)
                    print(f"[finalize] 新增 gap 子任务: [{gap_id}] {gap[:60]}")

            # 如果没有 gaps 但有 followup_queries，创建一个通用补查子任务
            if not new_gaps and next_queries:
                fallback_id = f"followup_{iteration}"
                if fallback_id not in existing_ids:
                    updated_subtasks.append({
                        "id": fallback_id,
                        "title": "Follow-up 补充检索",
                        "reason": "finalize 认为需要补充检索",
                        "queries": next_queries,
                        "depends_on": [],
                    })
                    print(f"[finalize] 新增 fallback 子任务: [{fallback_id}]")

            # 重算 parallel_groups：已完成的子任务会被 execute_subtasks 自动跳过
            from .parse_claims import _compute_parallel_groups
            new_groups = _compute_parallel_groups(updated_subtasks)

            result["subtasks"] = updated_subtasks
            result["parallel_groups"] = new_groups
            result["subtask_findings"] = updated_findings

            print(f"[finalize] followup: total_subtasks={len(updated_subtasks)}, "
                  f"completed={len(updated_findings)}, "
                  f"new_gaps={len(new_gaps)}, groups={len(new_groups)}")

        return result

    return finalize
