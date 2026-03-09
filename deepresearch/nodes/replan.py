# -*- coding: utf-8 -*-
"""
Replan controller.

职责：
- 根据 finalize 的 gaps / followup_queries 决定是否改计划
- 将 replanning 从 answer synthesis 中剥离出来
"""

from __future__ import annotations

from typing import Callable, Dict, List

from langchain_core.messages import AIMessage

from ..state import DeepResearchState
from .parse_claims import _compute_parallel_groups


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


def make_replan_node() -> Callable[[DeepResearchState], DeepResearchState]:
    async def replan(state: DeepResearchState) -> DeepResearchState:
        print("\n============ replan 阶段 ============")

        needs_followup = bool(state.get("needs_followup", False))
        iteration = int(state.get("iteration", 0))
        subtasks = list(state.get("subtasks", []) or [])
        subtask_findings = dict(state.get("subtask_findings", {}) or {})
        next_queries = _dedup_strings(state.get("queries", []) or [])
        new_gaps = _dedup_strings(state.get("research_gaps", []) or [])
        plan_review = dict(state.get("plan_review", {}) or {})

        if not needs_followup:
            print("[replan] finalize 已判定无需补充检索。")
            return {"messages": [AIMessage(content="[replan] 无需改计划。")]}

        updated_subtasks = list(subtasks)
        existing_ids = {str(st.get("id", "")).strip() for st in updated_subtasks}

        for gi, gap in enumerate(new_gaps):
            gap_id = f"gap_{iteration}_{gi}"
            if gap_id in existing_ids:
                continue
            q_per_gap = max(1, len(next_queries) // max(1, len(new_gaps)))
            start_idx = gi * q_per_gap
            gap_queries = next_queries[start_idx:start_idx + q_per_gap] or next_queries[:2]
            updated_subtasks.append({
                "id": gap_id,
                "title": f"补查: {gap[:60]}",
                "reason": gap,
                "queries": gap_queries,
                "depends_on": [],
                "guess_answer": "",
            })
            existing_ids.add(gap_id)
            print(f"[replan] 新增 gap 子任务: [{gap_id}] {gap[:60]}")

        if not new_gaps and next_queries:
            fallback_id = f"followup_{iteration}"
            if fallback_id not in existing_ids:
                updated_subtasks.append({
                    "id": fallback_id,
                    "title": "Follow-up 补充检索",
                    "reason": "replan 根据 finalize followup_queries 生成的补查任务",
                    "queries": next_queries,
                    "depends_on": [],
                    "guess_answer": "",
                })
                print(f"[replan] 新增 fallback 子任务: [{fallback_id}]")

        new_groups = _compute_parallel_groups(updated_subtasks)
        if str(plan_review.get("group_strategy", "")).lower() == "serialize":
            new_groups = [[str(st.get("id", "")).strip()] for st in updated_subtasks if str(st.get("id", "")).strip()]

        msg = AIMessage(
            content=(
                f"[replan] 追加子任务 {max(0, len(updated_subtasks) - len(subtasks))} 个，"
                f"并行组 {len(new_groups)} 层。"
            )
        )

        return {
            "subtasks": updated_subtasks,
            "parallel_groups": new_groups,
            "subtask_findings": subtask_findings,
            "queries": next_queries,
            "messages": [msg],
        }

    return replan
