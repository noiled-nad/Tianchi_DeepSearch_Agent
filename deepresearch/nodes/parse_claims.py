"""deepresearch/nodes/parse_claims.py

OAgents 风格子任务拆解：
1) 提炼研究简报（research_brief）
2) 将问题拆解为多个子任务（subtasks），标注依赖关系
3) 计算并行执行组（parallel_groups）
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Set

from langchain_core.messages import AIMessage, BaseMessage

from ..state import DeepResearchState
from ..plan_tips import get_plan_tips, format_tips_for_prompt
from ..prompt_loader import load_prompt


async def _ainvoke_with_stream_debug(llm, prompt: str, tag: str = "parse_claims") -> str:
    """
    调试用：优先走 astream 流式打印模型输出，便于观察卡点。
    若模型不支持流式，则退回 ainvoke。
    """
    t0 = time.perf_counter()
    print(f"[{tag}] request_sent, prompt_len={len(prompt)}")

    # 优先尝试流式
    if hasattr(llm, "astream"):
        chunks: List[str] = []
        first_token_at = None
        chunk_count = 0
        try:
            async for chunk in llm.astream(prompt):
                text = str(getattr(chunk, "content", "") or "")
                if not text:
                    continue
                chunk_count += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    print(f"[{tag}] first_token_latency={first_token_at - t0:.2f}s")
                    print(f"[{tag}] stream_start >>>")
                print(text, end="", flush=True)
                chunks.append(text)

            if first_token_at is not None:
                print(f"\n[{tag}] <<< stream_end")
            t1 = time.perf_counter()
            total = t1 - t0
            out_text = "".join(chunks)
            if first_token_at is None:
                print(f"[{tag}] no_stream_chunk_received, total_latency={total:.2f}s")
                print(f"[{tag}] diagnosis=请求/排队慢（首包未返回）")
            else:
                first = first_token_at - t0
                gen = t1 - first_token_at
                print(
                    f"[{tag}] timing_breakdown: request_wait={first:.2f}s, "
                    f"output_stream={gen:.2f}s, total={total:.2f}s, chunks={chunk_count}, out_len={len(out_text)}"
                )
                if first > 8 and gen < 3:
                    print(f"[{tag}] diagnosis=请求慢（首包等待长，输出本身快）")
                elif first < 3 and gen > 8:
                    print(f"[{tag}] diagnosis=输出慢（首包快，但生成耗时长）")
                else:
                    print(f"[{tag}] diagnosis=请求与输出均有耗时")
            return out_text
        except Exception as e:
            print(f"[{tag}] stream_failed, fallback_to_ainvoke: {e}")

    # 回退：非流式
    resp = await llm.ainvoke(prompt)
    t1 = time.perf_counter()
    total = t1 - t0
    out = str(getattr(resp, "content", "") or "")
    print(f"[{tag}] non_stream_done, total_latency={total:.2f}s, out_len={len(out)}")
    print(f"[{tag}] diagnosis=无法拆分首包与输出时间（非流式回退）")
    return out


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


SUBTASK_PLAN_PROMPT = load_prompt("parse_claims.yaml", "subtask_plan_prompt")


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
            raw = await _ainvoke_with_stream_debug(llm, prompt, tag="parse_claims")
            t_parse0 = time.perf_counter()
            obj = _safe_json_obj(raw)
            t_parse1 = time.perf_counter()
            print(f"[parse_claims] json_parse_latency={t_parse1 - t_parse0:.3f}s")
        except Exception as e:
            print(f"[parse_claims] LLM parse failed: {e}")
            obj = {}

        # ── 解析子任务（不含 queries，查询在执行阶段生成） ──
        raw_subtasks = obj.get("subtasks", [])
        subtasks: List[Dict] = []

        for st in raw_subtasks:
            if not isinstance(st, dict):
                continue
            st_id = str(st.get("id", f"ST{len(subtasks)+1}")).strip()
            title = str(st.get("title", "")).strip()
            reason = str(st.get("reason", "")).strip()
            depends_on = [str(d).strip() for d in st.get("depends_on", []) if str(d).strip()]

            subtasks.append({
                "id": st_id,
                "title": title,
                "reason": reason,
                "queries": [],  # 查询延迟到执行阶段生成
                "depends_on": depends_on,
                "guess_answer": str(st.get("guess_answer", "")).strip(),
            })

        # 兜底：如果没解析出子任务，创建单一子任务
        if not subtasks:
            subtasks = [{
                "id": "ST1",
                "title": question[:80],
                "reason": "直接搜索回答",
                "queries": [],
                "depends_on": [],
            }]

        # 计算并行组
        parallel_groups = _compute_parallel_groups(subtasks)

        # 研究简报
        brief = {
            "objective": str(obj.get("objective", "根据证据回答用户问题")).strip(),
            "answer_format": str(obj.get("answer_format", "简洁文本答案")).strip(),
            "problem_type": str(obj.get("problem_type", "entity_chain")).strip(),
            "hard_constraints": [str(x).strip() for x in obj.get("hard_constraints", []) if str(x).strip()],
            "key_entities": [str(x).strip() for x in obj.get("key_entities", []) if str(x).strip()],
            "done_criteria": [str(x).strip() for x in obj.get("done_criteria", []) if str(x).strip()],
        }

        # 打印规划结果
        print(f"[parse_claims] subtasks={len(subtasks)}, parallel_groups={parallel_groups}")
        print(f"[parse_claims] problem_type={brief.get('problem_type', 'unknown')}, answer_format={brief.get('answer_format', '')}")
        print(f"[parse_claims] hard_constraints={brief.get('hard_constraints', [])}")
        for st in subtasks:
            deps = st['depends_on']
            guess = st.get('guess_answer', '')
            print(f"  [{st['id']}] {st['title']}  (deps={deps})")
            print(f"       reason: {st['reason']}")
            if guess:
                print(f"       🔮 guess: {guess}")

        progress_msg = AIMessage(
            content=f"[plan] 拆解为 {len(subtasks)} 个子任务，"
                    f"并行组 {len(parallel_groups)} 层。"
                    f"查询将在执行阶段根据上下文动态生成。"
        )

        return {
            "question": question,
            "research_brief": brief,
            "plan_review": {},
            "subtasks": subtasks,
            "parallel_groups": parallel_groups,
            "subtask_findings": {},
            "documents": [],
            "execution_memory": {},
            "task_packets": {},
            "worker_artifacts": {},
            "queries": [],
            "query_history": [],
            "research_gaps": [],
            "final_answer": "",
            "needs_followup": True,
            "iteration": 0,
            "max_iterations": int(state.get("max_iterations", 4)),
            "messages": [progress_msg],
        }

    return parse_claims
