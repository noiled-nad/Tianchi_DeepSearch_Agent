# -*- coding: utf-8 -*-
"""
Context manager for orchestrator-driven task packets.

目标：
- 为每个 subtask 构造显式的 Task Packet
- 只传递任务相关的压缩上下文，而不是整个执行历史
- 为 candidate branch 构造更小的验证包
"""

from __future__ import annotations

from typing import Any, Dict, List

from .memory import ExecutionMemory


def _dedup_strings(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _guidance_for_subtask(plan_review: Dict[str, Any] | None, subtask_id: str) -> Dict[str, Any]:
    if not plan_review:
        return {}
    for item in plan_review.get("subtask_guidance", []) or []:
        if str(item.get("id", "")).strip() == subtask_id:
            return item
    return {}


def _summarize_dependencies(subtask: Dict[str, Any], prev_results: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for dep_id in subtask.get("depends_on", []) or []:
        dep = prev_results.get(dep_id) or {}
        if not isinstance(dep, dict):
            continue
        candidates = [str(c).strip() for c in dep.get("candidates", []) if str(c).strip()]
        evidence = [str(e).strip() for e in dep.get("evidence", []) if str(e).strip()]
        if candidates:
            line = f"[{dep_id}] 候选: {', '.join(candidates[:3])}"
        else:
            line = f"[{dep_id}] 候选: (未确定)"
        if evidence:
            line += f" | 证据: {'; '.join(evidence[:2])}"
        lines.append(line)
    return lines


def build_task_packet(
    question: str,
    brief: Dict[str, Any],
    subtask: Dict[str, Any],
    prev_results: Dict[str, Dict[str, Any]],
    memory: ExecutionMemory | None = None,
    plan_review: Dict[str, Any] | None = None,
    retry_context: str = "",
) -> Dict[str, Any]:
    guidance = _guidance_for_subtask(plan_review, subtask.get("id", ""))
    hard_constraints = [str(x).strip() for x in brief.get("hard_constraints", []) if str(x).strip()]
    local_constraints = [str(x).strip() for x in guidance.get("local_constraints", []) if str(x).strip()]
    constraints = _dedup_strings(hard_constraints + local_constraints)

    known_facts = _summarize_dependencies(subtask, prev_results)

    memory_context = ""
    if memory:
        memory_context = memory.to_context_for_subtask(subtask, detail_level="brief")

    curated_context_parts: List[str] = []
    instruction = str(guidance.get("instruction", "")).strip()
    context_focus = str(guidance.get("context_focus", "")).strip()
    if instruction:
        curated_context_parts.append(f"执行指令: {instruction}")
    if context_focus:
        curated_context_parts.append(f"重点线索: {context_focus}")
    if known_facts:
        curated_context_parts.append("依赖事实:\n" + "\n".join(known_facts))
    if constraints:
        curated_context_parts.append("约束:\n" + "\n".join(f"- {c}" for c in constraints))
    if retry_context:
        curated_context_parts.append("重试诊断:\n" + retry_context)
    if memory_context:
        curated_context_parts.append(memory_context)

    budget_hint = str(guidance.get("budget_hint", "standard")).strip().lower()
    if budget_hint == "conservative":
        budget = {"max_queries": 3, "max_docs": 8, "verify_docs": 4}
    else:
        budget = {"max_queries": 5, "max_docs": 8, "verify_docs": 5}

    allowed_tools = guidance.get("allowed_tools") or ["search", "fetch", "compress", "extract", "verify"]

    return {
        "subtask_id": subtask.get("id", ""),
        "goal": subtask.get("title", ""),
        "reason": subtask.get("reason", ""),
        "instruction": instruction or subtask.get("reason", ""),
        "context_focus": context_focus,
        "constraints": constraints,
        "known_facts": known_facts,
        "curated_context": "\n\n".join([part for part in curated_context_parts if part]),
        "allowed_tools": allowed_tools,
        "budget": budget,
        "model_hint": "flash",
    }


def build_candidate_packet(task_packet: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    candidate_constraints = task_packet.get("constraints", [])
    candidate_context = task_packet.get("curated_context", "")
    branch_instruction = (
        f"仅验证候选「{candidate}」是否满足当前子任务要求。"
        "优先寻找客观百科信息、时间线、平台/身份/关系是否匹配。"
    )
    return {
        "subtask_id": task_packet.get("subtask_id", ""),
        "candidate": candidate,
        "goal": f"{task_packet.get('goal', '')}｜候选核验",
        "instruction": branch_instruction,
        "context_focus": task_packet.get("context_focus", ""),
        "constraints": candidate_constraints,
        "curated_context": (
            f"{branch_instruction}\n"
            f"候选: {candidate}\n"
            f"原子任务目标: {task_packet.get('goal', '')}\n"
            f"{candidate_context}"
        ).strip(),
        "allowed_tools": ["search", "fetch", "compress", "verify"],
        "budget": {
            "max_queries": min(4, int(task_packet.get("budget", {}).get("max_queries", 4))),
            "max_docs": min(5, int(task_packet.get("budget", {}).get("verify_docs", 5))),
        },
        "model_hint": task_packet.get("model_hint", "flash"),
    }
