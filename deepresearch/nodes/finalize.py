"""节点：finalize

子任务汇总版：
1) 汇总所有子任务的 findings，结合原始文档生成答案
2) 判断是否需要继续检索
3) 如需继续，产出 follow-up queries
4) 反思本轮执行质量，生成改进建议（方向二）
5) 集成 ResearchMemory 记录执行轨迹（方向三）
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage

from ..schemas import Document
from ..state import DeepResearchState
from ..memory import ResearchMemory, ResearchStep, create_step


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


def _get_findings_text(findings_data: Any) -> str:
    """从 subtask_findings 中获取 findings 文本（兼容新旧格式）。"""
    if isinstance(findings_data, dict):
        return findings_data.get("findings", "")
    return str(findings_data) if findings_data else ""


def _format_subtask_findings(subtask_findings: Dict[str, Any], subtasks: List[Dict]) -> str:
    """将子任务发现格式化为结构化文本（兼容 SubtaskResult.to_dict() 格式）。"""
    if not subtask_findings:
        return ""
    lines = ["## 子任务调查结果"]
    st_map = {st["id"]: st for st in subtasks} if subtasks else {}
    for st_id, result_data in subtask_findings.items():
        st = st_map.get(st_id, {})
        title = st.get("title", st_id)

        # 兼容新旧格式
        if isinstance(result_data, dict):
            findings = result_data.get("findings", "")
            duration = result_data.get("duration_ms", 0)
            llm_calls = result_data.get("llm_calls", 0)
            success = result_data.get("success", True)
            meta_info = f" (耗时:{duration:.0f}ms, LLM调用:{llm_calls}次)" if duration else ""
            if not success:
                error = result_data.get("error", "未知错误")
                lines.append(f"\n### [{st_id}] {title}{meta_info} ❌ 失败")
                lines.append(f"错误: {error}")
            else:
                lines.append(f"\n### [{st_id}] {title}{meta_info}")
                lines.append(findings)
        else:
            lines.append(f"\n### [{st_id}] {title}")
            lines.append(str(result_data))
    return "\n".join(lines)


def _build_reflection_prompt(
    subtask_findings: Dict[str, Any],
    subtasks: List[Dict],
    iteration: int,
) -> str:
    """构建反思 prompt。"""
    # 统计成功/失败的子任务
    success_count = 0
    failed_count = 0
    no_info_count = 0
    total_llm_calls = 0
    total_duration = 0

    failed_subtasks = []
    no_info_subtasks = []

    for st_id, result_data in subtask_findings.items():
        if isinstance(result_data, dict):
            success = result_data.get("success", True)
            findings = result_data.get("findings", "")
            llm_calls = result_data.get("llm_calls", 0)
            duration = result_data.get("duration_ms", 0)

            total_llm_calls += llm_calls
            total_duration += duration

            if not success:
                failed_count += 1
                failed_subtasks.append({
                    "id": st_id,
                    "error": result_data.get("error", "未知"),
                })
            elif "未找到" in findings or "未找到相关信息" in findings:
                no_info_count += 1
                no_info_subtasks.append(st_id)
            else:
                success_count += 1

    total = success_count + failed_count + no_info_count
    success_rate = success_count / total if total > 0 else 0

    prompt = f"""请对第 {iteration + 1} 轮研究执行进行反思评估。

## 执行统计
- 成功子任务: {success_count}/{total} ({success_rate*100:.0f}%)
- 失败子任务: {failed_count}
- 无信息子任务: {no_info_count}
- 总 LLM 调用: {total_llm_calls} 次
- 总耗时: {total_duration/1000:.1f} 秒

## 失败详情
{json.dumps(failed_subtasks, ensure_ascii=False, indent=2) if failed_subtasks else "无"}

## 无信息子任务
{json.dumps(no_info_subtasks, ensure_ascii=False) if no_info_subtasks else "无"}

请输出 JSON（不要 markdown）：
{{
  "progress_score": 0.0-1.0,
  "strengths": ["本轮做得好的地方"],
  "weaknesses": ["本轮存在的问题"],
  "failed_subtasks_analysis": ["失败原因分析"],
  "suggestions": ["下一轮改进建议"],
  "query_strategy": "建议的查询策略调整"
}}

规则：
- progress_score 评估本轮整体进展（0=无进展，1=完全解决）
- suggestions 应具体可执行，如"尝试用中文搜索"、"换用更精确的关键词"
- query_strategy 总结查询优化的方向
"""
    return prompt


def _parse_reflection_response(raw: str) -> Dict[str, Any]:
    """解析反思响应。"""
    try:
        obj = _safe_json_obj(raw)
        return {
            "progress_score": float(obj.get("progress_score", 0.5)),
            "strengths": list(obj.get("strengths", [])),
            "weaknesses": list(obj.get("weaknesses", [])),
            "failed_subtasks_analysis": list(obj.get("failed_subtasks_analysis", [])),
            "suggestions": list(obj.get("suggestions", [])),
            "query_strategy": str(obj.get("query_strategy", "")),
        }
    except Exception:
        return {
            "progress_score": 0.5,
            "strengths": [],
            "weaknesses": [],
            "failed_subtasks_analysis": [],
            "suggestions": [],
            "query_strategy": "",
        }


def make_finalize_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def finalize(state: DeepResearchState) -> DeepResearchState:
        step_start = time.time()
        print("\n============ finalize 阶段 ============")

        question: str = state.get("question", "")
        brief = state.get("research_brief", {}) or {}
        docs: List[Document] = state.get("documents", []) or []
        subtask_findings = state.get("subtask_findings", {}) or {}
        subtasks = state.get("subtasks", []) or []
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 4))
        gaps = state.get("research_gaps", []) or []
        reflections = list(state.get("reflections", []) or [])

        # ── 恢复 ResearchMemory ──
        memory_data = state.get("memory")
        if memory_data:
            memory = ResearchMemory.from_dict(memory_data)
        else:
            memory = ResearchMemory()

        print(f"[finalize] iteration={iteration}/{max_iterations}, "
              f"documents={len(docs)}, subtask_findings={len(subtask_findings)}, gaps={len(gaps)}")

        # 构建证据：子任务发现 + 原始文档
        findings_text = _format_subtask_findings(subtask_findings, subtasks)
        if subtask_findings:
            sources_text = _format_sources_index(docs)
            sources_label = "文档引用索引"
        else:
            sources_text = _format_sources_full(docs)
            sources_label = "原始证据包"

        # 构建前序反思上下文（如果有）
        prev_suggestions = []
        if reflections:
            for r in reflections[-2:]:  # 最近2轮的建议
                prev_suggestions.extend(r.get("suggestions", []))
            prev_suggestions = prev_suggestions[-5:]  # 最多5条

        # 从 memory 获取前序步骤摘要
        memory_context = ""
        if memory.steps:
            recent_messages = memory.to_messages(summary_mode=True, max_steps=5)
            if recent_messages:
                memory_context = "\n\n## 前序执行摘要\n" + "\n".join(
                    m.get("content", "") for m in recent_messages
                )

        reflection_context = ""
        if prev_suggestions:
            reflection_context = "\n\n## 前序反思建议（请参考）\n" + "\n".join(f"- {s}" for s in prev_suggestions)

        prompt = (
            "基于子任务调查结果和证据，输出 JSON（不要 markdown code fence）：\n"
            "{\"reasoning\":string,\"final_answer\":string,\"confidence\":number,"
            "\"needs_followup\":boolean,\"research_gaps\":[string],\"followup_queries\":[string]}\n"
            "规则：reasoning 简洁串联推理链；final_answer 直接可交付，用[S1]引用来源；"
            "证据不足时 needs_followup=true 给3~6条followup_queries。\n\n"
            f"问题：{question}\n"
            f"已知缺口：{json.dumps(gaps, ensure_ascii=False)}\n\n"
            f"{findings_text}\n\n"
            f"{sources_label}：\n{sources_text}\n"
            f"{reflection_context}"
            f"{memory_context}"
        )

        print(f"[finalize] prompt_len={len(prompt)} chars")
        resp = await llm.ainvoke(prompt, max_tokens=2048)
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

        # ── 反思评估（方向二） ──
        reflection_prompt = _build_reflection_prompt(subtask_findings, subtasks, iteration)
        reflection_data = {
            "iteration": iteration,
            "progress_score": 0.5,
            "suggestions": [],
        }
        try:
            reflection_resp = await llm.ainvoke(reflection_prompt, max_tokens=1000)
            reflection_data = _parse_reflection_response(str(reflection_resp.content))
            reflection_data["iteration"] = iteration
            reflections.append(reflection_data)
            print(f"[finalize] reflection: progress_score={reflection_data['progress_score']:.2f}, "
                  f"suggestions={len(reflection_data['suggestions'])}")
        except Exception as e:
            print(f"[finalize] reflection failed: {e}")
            reflections.append(reflection_data)

        step_end = time.time()

        # ── 创建 finalize 的 ResearchStep ──
        finalize_step = create_step(
            step_type="finalize",
            step_id=f"finalize_{iteration}",
            input_summary=f"迭代 {iteration + 1}: 汇总 {len(subtask_findings)} 个子任务",
            output_summary=answer_text[:200] if answer_text else "无",
            success=True,
            score=reflection_data.get("progress_score"),
            reflection=json.dumps(reflection_data.get("suggestions", []), ensure_ascii=False),
        )
        finalize_step.start_time = step_start
        finalize_step.end_time = step_end
        finalize_step.duration_ms = (step_end - step_start) * 1000
        memory.add_step(finalize_step)

        # ── 如果有反思，也创建一个 reflect 步骤 ──
        if reflection_data.get("suggestions"):
            reflect_step = create_step(
                step_type="reflect",
                step_id=f"reflect_{iteration}",
                input_summary=f"反思第 {iteration + 1} 轮执行",
                output_summary=f"进度评分: {reflection_data['progress_score']:.2f}, "
                              f"建议: {len(reflection_data['suggestions'])} 条",
                success=True,
                score=reflection_data.get("progress_score"),
            )
            memory.add_step(reflect_step)

        progress = AIMessage(
            content=(
                f"[finalize] 置信度={obj.get('confidence', 0)}，"
                f"needs_followup={need_followup}，下一轮查询={len(next_queries) if need_followup else 0}。"
            )
        )

        result = {
            "final_answer": answer_text,
            "needs_followup": need_followup,
            "research_gaps": new_gaps,
            "queries": next_queries if need_followup else [],
            "iteration": iteration + 1,
            "reflections": reflections,
            "memory": memory.to_dict(),  # 保存 memory
            "messages": [progress, AIMessage(content=answer_text)],
        }

        # ── followup 策略：保留原子任务 + 为 gaps 新增定向子任务 ──
        if need_followup:
            updated_subtasks = list(subtasks) if subtasks else []
            updated_findings = dict(subtask_findings)
            existing_ids = {st["id"] for st in updated_subtasks}

            for gi, gap in enumerate(new_gaps):
                gap_id = f"gap_{iteration}_{gi}"
                if gap_id not in existing_ids:
                    q_per_gap = max(1, len(next_queries) // max(1, len(new_gaps)))
                    start_idx = gi * q_per_gap
                    gap_queries = next_queries[start_idx:start_idx + q_per_gap]
                    if not gap_queries:
                        gap_queries = next_queries[:2] if next_queries else []

                    # 将反思建议注入 gap 子任务的 reason
                    suggestions_text = ""
                    if reflections and reflections[-1].get("suggestions"):
                        suggestions_text = f" | 建议: {reflections[-1]['suggestions'][0]}"

                    updated_subtasks.append({
                        "id": gap_id,
                        "title": f"补查: {gap[:60]}",
                        "reason": gap + suggestions_text,
                        "queries": gap_queries,
                        "depends_on": [],
                    })
                    existing_ids.add(gap_id)
                    print(f"[finalize] 新增 gap 子任务: [{gap_id}] {gap[:60]}")

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

            from .parse_claims import _compute_parallel_groups
            new_groups = _compute_parallel_groups(updated_subtasks)

            result["subtasks"] = updated_subtasks
            result["parallel_groups"] = new_groups
            result["subtask_findings"] = updated_findings

            print(f"[finalize] followup: total_subtasks={len(updated_subtasks)}, "
                  f"completed={len(updated_findings)}, "
                  f"new_gaps={len(new_gaps)}, groups={len(new_groups)}")

        # 打印 memory 统计
        stats = memory.get_statistics()
        print(f"[finalize] memory stats: {stats['total_steps']} steps, "
              f"success_rate={stats['success_rate']*100:.0f}%")

        return result

    return finalize
