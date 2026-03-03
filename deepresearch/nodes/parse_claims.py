"""deepresearch/nodes/parse_claims.py

OAgents 风格子任务拆解：
1) 提炼研究简报（research_brief）
2) 将问题拆解为多个子任务（subtasks），标注依赖关系
3) 计算并行执行组（parallel_groups）
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Set

from langchain_core.messages import AIMessage, BaseMessage

from ..state import DeepResearchState
from ..plan_tips import get_plan_tips, format_tips_for_prompt


def _extract_last_user_question(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            return str(m.content).strip()
    return str(messages[-1].content).strip() if messages else ""


def _safe_json_obj(text: str) -> Dict:
    t = text.strip()
    # 去掉 markdown code fence
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


def _fallback_queries(question: str) -> List[str]:
    q = question.strip()
    if not q:
        return ["deep research answer"]
    return [q, f"{q} 官方 资料", f"{q} wikipedia"]


def _compute_parallel_groups(subtasks: List[Dict]) -> List[List[str]]:
    """
    根据 depends_on 拓扑排序，产出并行执行组。
    同一组内的子任务互不依赖，可并行。
    """
    if not subtasks:
        return []

    remaining = {
        st["id"]: set(st.get("depends_on", []))
        for st in subtasks
    }
    groups: List[List[str]] = []

    while remaining:
        # 找出所有依赖已满足（空集）的任务
        ready = [sid for sid, deps in remaining.items() if not deps]
        if not ready:
            # 有环或残留，全部放一组兜底
            ready = list(remaining.keys())
        groups.append(sorted(ready))
        for sid in ready:
            del remaining[sid]
        # 从剩余任务的依赖中移除已完成的
        for deps in remaining.values():
            deps -= set(ready)

    return groups


SUBTASK_PLAN_PROMPT = """\
你是 Deep Research 的任务规划器。请将用户问题拆解为可独立搜索执行的子任务。

## 拆解规则
1) 每个子任务应聚焦一个独立的信息检索目标
2) 多跳问题必须拆成链式子任务（ST1 的结果用于 ST2 的查询）
3) 可并行的子任务不要标注依赖
4) 每个子任务给 2~4 条搜索查询（短查询，覆盖中英双语，聚焦年份/人名/作品名等锚点）
5) 子任务数量通常 2~5 个，简单问题可以只有 1 个

## 查询编写规则（极重要）
- 每条查询只检索一个实体/概念，绝不在一条查询中混合多个检索目标
- 查询要短（3~8 个词），聚焦年份/人名/作品名等锚点词
- 覆盖中英双语（同一目标的中文和英文各一条）
- 对于依赖前序子任务的子任务，queries 写成"单侧检索"形式：
  例如子任务是"找出A和B的共同点"，queries 应分别查 A 的信息和 B 的信息，
  而不是把 A 和 B 塞进同一条查询

## 输出格式（严格 JSON，不要 markdown）
{{
  "objective": "一句话研究目标",
  "answer_format": "期望的答案格式（如：人名/年份/数字...）",
  "key_entities": ["实体1", "实体2"],
  "hard_constraints": ["约束1"],
  "done_criteria": ["完成标准1"],
  "subtasks": [
    {{
      "id": "ST1",
      "title": "子任务标题（简洁明确）",
      "reason": "为什么需要这个子任务",
      "queries": ["搜索词1", "搜索词2"],
      "depends_on": []
    }},
    {{
      "id": "ST2",
      "title": "子任务标题",
      "reason": "需要 ST1 的结果来确定...",
      "queries": ["搜索词1", "搜索词2"],
      "depends_on": ["ST1"]
    }}
  ]
}}
{tips_block}
问题：{question}"""


def make_parse_claims_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def parse_claims(state: DeepResearchState) -> DeepResearchState:
        print("\n============ parse_claims(subtask planning) 阶段 ============")
        question = state.get("question") or _extract_last_user_question(state.get("messages", []))
        print(f"[parse_claims] question_len={len(question)}")

        # OAgents Plan Tips：根据问题特征注入经验规则
        tips = get_plan_tips(question)
        tips_block = format_tips_for_prompt(tips)
        if tips_block:
            tips_block = "\n\n" + tips_block + "\n"
            print(f"[parse_claims] plan_tips matched: {len(tips)} tips")
        else:
            tips_block = ""
            print("[parse_claims] plan_tips matched: 0 tips")

        prompt = SUBTASK_PLAN_PROMPT.format(
            question=question,
            tips_block=tips_block,
        )

        try:
            resp = await llm.ainvoke(prompt)
            obj = _safe_json_obj(str(resp.content))
        except Exception as e:
            print(f"[parse_claims] LLM parse failed: {e}")
            obj = {}

        # ── 解析子任务 ──
        raw_subtasks = obj.get("subtasks", [])
        subtasks: List[Dict] = []
        all_queries: List[str] = []

        for st in raw_subtasks:
            if not isinstance(st, dict):
                continue
            st_id = str(st.get("id", f"ST{len(subtasks)+1}")).strip()
            title = str(st.get("title", "")).strip()
            reason = str(st.get("reason", "")).strip()
            queries = [str(q).strip() for q in st.get("queries", []) if str(q).strip()]
            depends_on = [str(d).strip() for d in st.get("depends_on", []) if str(d).strip()]

            if not queries and title:
                queries = [title]  # 兜底：用标题做查询

            subtasks.append({
                "id": st_id,
                "title": title,
                "reason": reason,
                "queries": queries,
                "depends_on": depends_on,
            })
            all_queries.extend(queries)

        # 兜底：如果没解析出子任务，创建单一子任务
        if not subtasks:
            fallback_queries = _fallback_queries(question)
            subtasks = [{
                "id": "ST1",
                "title": question[:80],
                "reason": "直接搜索回答",
                "queries": fallback_queries,
                "depends_on": [],
            }]
            all_queries = fallback_queries

        # 计算并行组
        parallel_groups = _compute_parallel_groups(subtasks)

        # 研究简报
        brief = {
            "objective": str(obj.get("objective", "根据证据回答用户问题")).strip(),
            "answer_format": str(obj.get("answer_format", "简洁文本答案")).strip(),
            "hard_constraints": [str(x).strip() for x in obj.get("hard_constraints", []) if str(x).strip()],
            "key_entities": [str(x).strip() for x in obj.get("key_entities", []) if str(x).strip()],
            "done_criteria": [str(x).strip() for x in obj.get("done_criteria", []) if str(x).strip()],
        }

        # 打印规划结果
        print(f"[parse_claims] subtasks={len(subtasks)}, parallel_groups={parallel_groups}")
        for st in subtasks:
            deps = st['depends_on']
            print(f"  [{st['id']}] {st['title']}  (queries={len(st['queries'])}, deps={deps})")
            for q in st['queries']:
                print(f"       -> {q}")

        progress_msg = AIMessage(
            content=f"[plan] 拆解为 {len(subtasks)} 个子任务，"
                    f"并行组 {len(parallel_groups)} 层，"
                    f"总计 {len(all_queries)} 条初始查询。"
        )

        return {
            "question": question,
            "research_brief": brief,
            "subtasks": subtasks,
            "parallel_groups": parallel_groups,
            "subtask_findings": {},
            "queries": all_queries,
            "query_history": all_queries[:],
            "research_gaps": [],
            "needs_followup": True,
            "iteration": int(state.get("iteration", 0)),
            "max_iterations": int(state.get("max_iterations", 4)),
            "messages": [progress_msg],
        }

    return parse_claims
