# -*- coding: utf-8 -*-
"""
Pre-execution plan review.

职责：
- 审查 planner 输出的 subtasks / groups 是否存在明显风险
- 补齐缺失约束
- 为后续 worker 构造更明确的执行指导
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List

from langchain_core.messages import AIMessage

from ..prompt_loader import load_prompt
from ..state import DeepResearchState


PLAN_REVIEW_PROMPT = load_prompt("orchestration.yaml", "plan_review_prompt")


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


def _serialize_groups(subtasks: List[Dict[str, str]]) -> List[List[str]]:
    return [[str(st.get("id", "")).strip()] for st in subtasks if str(st.get("id", "")).strip()]


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


def make_review_plan_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def review_plan(state: DeepResearchState) -> DeepResearchState:
        print("\n============ review_plan 阶段 ============")

        question = str(state.get("question", "")).strip()
        brief = dict(state.get("research_brief", {}) or {})
        subtasks = list(state.get("subtasks", []) or [])
        parallel_groups = list(state.get("parallel_groups", []) or [])

        if not subtasks:
            print("[review_plan] 无子任务，跳过。")
            return {"messages": [AIMessage(content="[review_plan] 无子任务，跳过。")]}

        prompt = PLAN_REVIEW_PROMPT.format(
            question=question,
            brief_json=json.dumps(brief, ensure_ascii=False, indent=2),
            subtasks_json=json.dumps(subtasks, ensure_ascii=False, indent=2),
            parallel_groups_json=json.dumps(parallel_groups, ensure_ascii=False),
        )

        plan_review: Dict[str, object] = {
            "assessment": "",
            "plan_ok": True,
            "group_strategy": "keep",
            "missing_constraints": [],
            "global_risks": [],
            "subtask_guidance": [],
        }

        try:
            resp = await llm.ainvoke(prompt)
            obj = _safe_json_obj(str(resp.content).strip())
            plan_review["assessment"] = str(obj.get("assessment", "")).strip()
            plan_review["plan_ok"] = bool(obj.get("plan_ok", True))
            plan_review["group_strategy"] = str(obj.get("group_strategy", "keep")).strip().lower() or "keep"
            plan_review["missing_constraints"] = _dedup_strings(obj.get("missing_constraints", []) or [])
            plan_review["global_risks"] = _dedup_strings(obj.get("global_risks", []) or [])
            raw_guidance = obj.get("subtask_guidance", []) or []
            if isinstance(raw_guidance, list):
                plan_review["subtask_guidance"] = raw_guidance
        except Exception as e:
            print(f"[review_plan] FAILED: {type(e).__name__}: {e}")
            plan_review["assessment"] = "计划审查失败，保留原始规划并继续执行。"

        covered_ids = {
            str(item.get("id", "")).strip()
            for item in (plan_review.get("subtask_guidance", []) or [])
            if isinstance(item, dict)
        }
        fallback_guidance = list(plan_review.get("subtask_guidance", []) or [])
        for st in subtasks:
            st_id = str(st.get("id", "")).strip()
            if not st_id or st_id in covered_ids:
                continue
            fallback_guidance.append({
                "id": st_id,
                "instruction": str(st.get("reason", "")).strip() or str(st.get("title", "")).strip(),
                "context_focus": str(st.get("title", "")).strip(),
                "local_constraints": [],
                "allowed_tools": ["search", "fetch", "compress", "extract", "verify"],
                "budget_hint": "standard",
            })
        plan_review["subtask_guidance"] = fallback_guidance

        updated_brief = dict(brief)
        updated_brief["hard_constraints"] = _dedup_strings(
            [str(x).strip() for x in brief.get("hard_constraints", []) if str(x).strip()]
            + [str(x).strip() for x in plan_review.get("missing_constraints", []) if str(x).strip()]
        )

        updated_groups = parallel_groups
        if plan_review.get("group_strategy") == "serialize":
            updated_groups = _serialize_groups(subtasks)

        assessment = str(plan_review.get("assessment", "")).strip()
        risks = plan_review.get("global_risks", []) or []
        print(f"[review_plan] group_strategy={plan_review.get('group_strategy')}, risks={len(risks)}")
        if assessment:
            print(f"[review_plan] assessment: {assessment}")
        if risks:
            print(f"[review_plan] global_risks: {risks}")

        msg = AIMessage(
            content=(
                f"[review_plan] plan_ok={plan_review.get('plan_ok', True)}，"
                f"group_strategy={plan_review.get('group_strategy', 'keep')}，"
                f"补充约束={len(plan_review.get('missing_constraints', []))}。"
            )
        )

        return {
            "research_brief": updated_brief,
            "parallel_groups": updated_groups,
            "plan_review": plan_review,
            "messages": [msg],
        }

    return review_plan
