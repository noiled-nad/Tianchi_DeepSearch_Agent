"""deepresearch/nodes/execute_subtasks.py

OAgents 风格子任务执行器：
按 parallel_groups 分层并行执行子任务，每个子任务内部走：
  query_optimize (batch reflect + parallel rollout)
  → search
  → fetch
  → extract_findings (LLM 抽取关键发现)

后序子任务会注入前序子任务的 findings 作为上下文，
实现级联推理（多跳问题的核心价值）。

方向三：集成 ResearchMemory 体系
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Dict, List, Set, Tuple

from langchain_core.messages import AIMessage

from ..schemas import Document, SearchResult, SubtaskResult
from ..state import DeepResearchState
from ..memory import ResearchMemory, ResearchStep, ToolCall, create_step, create_tool_call
from .query_optimize import _reflect_batch, _rollout_query


# ───────── Prompts ─────────

FINDING_EXTRACTION_PROMPT = """\
你是信息提取专家。请根据以下文档内容，提取与当前子任务相关的关键发现。

原始问题：{question}
子任务：[{subtask_id}] {subtask_title}
{deps_context}

文档内容：
{docs_text}

要求：
1) 只提取与子任务直接相关的事实（人名、时间、地点、关系、数字等）
2) 用简洁的要点列表输出（3~8 条）
3) 标注信息来源的文档编号 [D1] [D2] 等
4) 如果文档中没有相关信息，直接回答"未找到相关信息"
5) 不要编造任何信息

关键发现："""


DEPS_QUERY_REFINE_PROMPT = """\
根据前序子任务的发现，为当前子任务优化搜索查询。

原始问题：{question}
当前子任务：{subtask_title}
原始查询：{original_queries}

前序子任务发现：
{deps_findings}

## 关键规则
1) 每条查询只检索一个实体/概念/事实，绝不混合多个检索目标
2) 用前序发现中的具体实体名替代原始查询中的泛化描述
3) 如果子任务需要"匹配"或"交叉对比"两侧信息，必须为每一侧分别生成独立查询
   - 错误示例："A产品特性 B产品特性 对比"（混合多目标，搜索引擎无法理解）
   - 正确示例：分别查 "A产品 特性列表"、"B产品 特性列表"
4) 生成 3~6 条短查询（每条 3~8 词），覆盖中英双语

只输出查询列表，每行一条，不要其他内容。"""


# ───────── 工具函数 ─────────

def _dedup_queries(queries: List[str], history: Set[str]) -> List[str]:
    """去重 + 排除已搜索过的。"""
    seen: Set[str] = set()
    result: List[str] = []
    for q in queries:
        q_clean = q.strip()
        q_norm = q_clean.lower()
        if not q_clean or q_norm in seen or q_norm in history:
            continue
        seen.add(q_norm)
        result.append(q_clean)
    return result


def _format_docs_text(docs: List[Document], max_chars_each: int = 1200) -> str:
    """格式化文档为文本，供 LLM 抽取。"""
    if not docs:
        return "（无文档）"
    chunks = []
    for i, d in enumerate(docs, 1):
        title = (d.title or "").strip().replace("\n", " ")
        content = d.content or ""
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "\n[截断]"
        chunks.append(f"[D{i}] {title}\nURL: {d.url}\n{content}")
    return "\n\n".join(chunks)


def _get_findings_text(findings_data: Any) -> str:
    """从 subtask_findings 中获取 findings 文本（兼容新旧格式）。"""
    if isinstance(findings_data, dict):
        return findings_data.get("findings", "")
    return str(findings_data) if findings_data else ""


# ───────── 子任务内部流程 ─────────

async def _optimize_queries_for_subtask(
    flash_llm,
    question: str,
    subtask: Dict,
    key_entities: List[str],
    query_history: List[str],
    deps_findings: str = "",
) -> Tuple[List[str], List[str], List[str]]:
    """
    为单个子任务优化查询：
    1) 如果有前序 findings，先用 LLM 基于 findings 细化查询
    2) batch reflect
    3) parallel rollout
    4) 去重

    返回: (final_queries, reflected_queries, rollout_queries)
    """
    raw_queries = list(subtask.get("queries", []))
    if not raw_queries:
        return [], [], []

    # ── 构建子任务聚焦上下文（不传完整问题，避免 reflect/rollout 跑偏到其他子任务） ──
    st_title = subtask.get("title", "")
    st_reason = subtask.get("reason", "")
    subtask_context = f"{st_title}。{st_reason}" if st_reason else st_title

    reflected_queries = []
    rollout_queries = []

    # 如果有依赖的前序发现，先用 LLM 细化查询
    if deps_findings:
        try:
            prompt = DEPS_QUERY_REFINE_PROMPT.format(
                question=question,
                subtask_title=st_title,
                original_queries="\n".join(f"- {q}" for q in raw_queries),
                deps_findings=deps_findings,
            )
            resp = await flash_llm.ainvoke(prompt)
            refined = [ln.strip().lstrip("- ·•0123456789.)") for ln in str(resp.content).split("\n") if ln.strip()]
            refined = [q for q in refined if len(q) > 3]
            if refined:
                raw_queries = refined + raw_queries
                print(f"  [{subtask['id']}] deps_refine: +{len(refined)} queries from findings")
        except Exception as e:
            print(f"  [{subtask['id']}] deps_refine failed: {e}")

    # Batch reflect（传子任务上下文而非完整问题）
    reflect_results = await _reflect_batch(flash_llm, subtask_context, raw_queries)
    reflected = [aug for (_, aug) in reflect_results]
    reflected_queries = list(reflected)

    # Parallel rollout（同样传子任务上下文）
    rollout_n = max(2, min(3, 8 // max(1, len(reflected))))

    async def _do_rollout(q: str) -> List[str]:
        return await _rollout_query(flash_llm, subtask_context, q, key_entities, query_history, rollout_n)

    rollout_results = await asyncio.gather(*[_do_rollout(q) for q in reflected])
    for variants in rollout_results:
        rollout_queries.extend(variants)

    # 重要：保留原始查询，避免优化阶段发生语义漂移
    all_queries = list(raw_queries) + list(reflected) + rollout_queries

    # 去重
    deduped = _dedup_queries(all_queries, {q.lower() for q in query_history})
    return deduped[:10], reflected_queries, rollout_queries  # 每个子任务最多 10 条优化查询


async def _search_and_fetch(
    queries: List[str],
    searcher,
    fetcher,
    existing_urls: Set[str],
    tool_calls: List[ToolCall],  # 用于记录工具调用
    max_docs: int = 6,
    per_query_results: int = 3,
    parallelism: int = 3,
) -> Tuple[List[Document], int]:
    """
    搜索 + 抓取，返回 (new_docs, search_results_count)。
    同时记录工具调用到 tool_calls。
    """
    seen_urls = set(existing_urls)
    candidate_items: List[Tuple[str, str]] = []  # (url, query)
    search_results_count = 0

    # Search
    for query in queries:
        if len(candidate_items) >= max_docs:
            break
        search_start = time.time()
        try:
            results: List[SearchResult] = await searcher.search(query)
            search_results_count += len(results)
            # 记录搜索工具调用
            tool_calls.append(create_tool_call(
                name="search",
                arguments={"query": query},
                result=f"找到 {len(results)} 条结果",
                duration_ms=(time.time() - search_start) * 1000,
            ))
        except Exception as e:
            tool_calls.append(create_tool_call(
                name="search",
                arguments={"query": query},
                error=str(e),
                duration_ms=(time.time() - search_start) * 1000,
            ))
            continue
        for r in results[:per_query_results]:
            if len(candidate_items) >= max_docs:
                break
            if not r.url or r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            candidate_items.append((r.url, query))

    if not candidate_items:
        return [], search_results_count

    # Fetch（并发）
    sem = asyncio.Semaphore(parallelism)

    async def _fetch_one(url: str, query: str):
        fetch_start = time.time()
        async with sem:
            try:
                try:
                    doc = await fetcher.fetch(url, query=query)
                    tool_calls.append(create_tool_call(
                        name="fetch",
                        arguments={"url": url, "query": query},
                        result=f"抓取成功: {doc.title or url[:50]}",
                        duration_ms=(time.time() - fetch_start) * 1000,
                    ))
                    return doc
                except TypeError:
                    doc = await fetcher.fetch(url)
                    tool_calls.append(create_tool_call(
                        name="fetch",
                        arguments={"url": url},
                        result=f"抓取成功: {doc.title or url[:50]}",
                        duration_ms=(time.time() - fetch_start) * 1000,
                    ))
                    return doc
            except Exception as e:
                tool_calls.append(create_tool_call(
                    name="fetch",
                    arguments={"url": url},
                    error=str(e),
                    duration_ms=(time.time() - fetch_start) * 1000,
                ))
                return None

    new_docs: List[Document] = []
    tasks = [asyncio.create_task(_fetch_one(u, q)) for (u, q) in candidate_items]
    for fut in asyncio.as_completed(tasks):
        doc = await fut
        if doc is not None:
            new_docs.append(doc)
        if len(new_docs) >= max_docs:
            break

    return new_docs, search_results_count


async def _extract_findings(
    flash_llm,
    question: str,
    subtask: Dict,
    docs: List[Document],
    deps_context: str = "",
) -> str:
    """用 LLM 从文档中提取与子任务相关的发现。"""
    if not docs:
        return "未找到相关文档。"

    docs_text = _format_docs_text(docs)
    prompt = FINDING_EXTRACTION_PROMPT.format(
        question=question,
        subtask_id=subtask["id"],
        subtask_title=subtask.get("title", ""),
        deps_context=deps_context,
        docs_text=docs_text,
    )

    try:
        resp = await flash_llm.ainvoke(prompt)
        findings = str(resp.content).strip()
        return findings if findings else "提取失败。"
    except Exception as e:
        print(f"  [{subtask['id']}] extract_findings failed: {e}")
        return "提取失败。"


async def _process_one_subtask(
    subtask: Dict,
    question: str,
    key_entities: List[str],
    query_history: List[str],
    deps_findings: str,
    flash_llm,
    searcher,
    fetcher,
    existing_urls: Set[str],
    memory: ResearchMemory,  # 新增：memory 参数
) -> Tuple[str, SubtaskResult]:
    """
    处理单个子任务完整流程：optimize → search → fetch → extract。
    返回 (subtask_id, SubtaskResult)。
    同时创建 ResearchStep 并添加到 memory。
    """
    st_id = subtask["id"]
    st_title = subtask.get("title", "")
    print(f"\n  ── 执行子任务 [{st_id}] {st_title} ──")

    step_start = time.time()
    llm_calls = 0
    error = None
    tool_calls: List[ToolCall] = []

    try:
        # 1) 优化查询
        deps_context = ""
        if deps_findings:
            deps_context = f"\n前序子任务发现：\n{deps_findings}"

        final_queries, reflected_queries, rollout_queries = await _optimize_queries_for_subtask(
            flash_llm, question, subtask, key_entities, query_history, deps_findings
        )
        llm_calls += 1  # reflect_batch
        llm_calls += len(reflected_queries)  # rollout calls

        print(f"  [{st_id}] optimized_queries={len(final_queries)}")
        for i, q in enumerate(final_queries, 1):
            print(f"  [{st_id}]   q{i}: {q}")

        if not final_queries:
            result = SubtaskResult(
                subtask_id=st_id,
                title=st_title,
                queries_original=subtask.get("queries", []),
                queries_final=[],
                findings="未生成有效查询。",
                success=False,
                error="No valid queries generated",
                duration_ms=(time.time() - step_start) * 1000,
                llm_calls=llm_calls,
            )
            # 创建失败的 ResearchStep
            step = create_step(
                step_type="subtask",
                step_id=st_id,
                input_summary=f"子任务: {st_title}",
                output_summary="未生成有效查询",
                tool_calls=tool_calls,
                success=False,
                error="No valid queries generated",
            )
            step.duration_ms = (time.time() - step_start) * 1000
            memory.add_step(step)
            return st_id, result

        # 2) 搜索 + 抓取
        new_docs, search_results_count = await _search_and_fetch(
            final_queries, searcher, fetcher, existing_urls, tool_calls
        )
        print(f"  [{st_id}] fetched_docs={len(new_docs)}")

        # 3) 抽取发现
        findings = await _extract_findings(
            flash_llm, question, subtask, new_docs, deps_context
        )
        llm_calls += 1
        print(f"  [{st_id}] findings: {findings[:120]}...")

        step_end = time.time()
        result = SubtaskResult(
            subtask_id=st_id,
            title=st_title,
            queries_original=subtask.get("queries", []),
            queries_reflected=reflected_queries,
            queries_rollout=rollout_queries,
            queries_final=final_queries,
            search_results_count=search_results_count,
            docs_fetched_urls=[d.url for d in new_docs if d.url],
            docs_fetched_count=len(new_docs),
            findings=findings,
            success=True,
            duration_ms=(step_end - step_start) * 1000,
            llm_calls=llm_calls,
        )

        # 创建成功的 ResearchStep
        step = create_step(
            step_type="subtask",
            step_id=st_id,
            input_summary=f"子任务: {st_title}, 查询: {len(final_queries)} 条",
            output_summary=findings[:200] if findings else "无",
            tool_calls=tool_calls,
            success=True,
        )
        step.start_time = step_start
        step.end_time = step_end
        step.duration_ms = (step_end - step_start) * 1000
        memory.add_step(step)

        return st_id, result

    except Exception as e:
        error = str(e)
        print(f"  [{st_id}] ERROR: {error}")
        step_end = time.time()
        result = SubtaskResult(
            subtask_id=st_id,
            title=st_title,
            queries_original=subtask.get("queries", []),
            success=False,
            error=error,
            duration_ms=(step_end - step_start) * 1000,
            llm_calls=llm_calls,
        )

        # 创建失败的 ResearchStep
        step = create_step(
            step_type="subtask",
            step_id=st_id,
            input_summary=f"子任务: {st_title}",
            output_summary="执行失败",
            tool_calls=tool_calls,
            success=False,
            error=error,
        )
        step.start_time = step_start
        step.end_time = step_end
        step.duration_ms = (step_end - step_start) * 1000
        memory.add_step(step)

        return st_id, result


# ───────── Graph Node ─────────

def make_execute_subtasks_node(
    llm,
    flash_llm,
    searcher,
    fetcher,
) -> Callable[[DeepResearchState], DeepResearchState]:
    """
    OAgents 风格子任务执行器。

    按 parallel_groups 分层执行：
    - 同层内 asyncio.gather 并行
    - 后序层可使用前序层的 findings
    - 集成 ResearchMemory 记录执行轨迹
    """

    async def execute_subtasks(state: DeepResearchState) -> DeepResearchState:
        print("\n============ execute_subtasks 阶段 ============")

        _flash = flash_llm or llm
        question: str = state.get("question", "")
        brief = state.get("research_brief", {}) or {}
        key_entities: List[str] = brief.get("key_entities", [])
        query_history: List[str] = list(state.get("query_history", []) or [])
        existing_docs: List[Document] = list(state.get("documents", []) or [])
        existing_urls: Set[str] = {d.url for d in existing_docs if getattr(d, "url", None)}

        # subtask_findings 现在存储 SubtaskResult.to_dict()
        subtask_findings: Dict[str, Any] = dict(state.get("subtask_findings", {}) or {})

        # ── 恢复或创建 ResearchMemory ──
        memory_data = state.get("memory")
        if memory_data:
            memory = ResearchMemory.from_dict(memory_data)
        else:
            memory = ResearchMemory()

        # ── 获取子任务列表 ──
        subtasks: List[Dict] = state.get("subtasks", []) or []
        parallel_groups: List[List[str]] = state.get("parallel_groups", []) or []

        # 兜底：如果没有子任务但有 queries（followup 场景），构造单一子任务
        if not subtasks and state.get("queries"):
            subtasks = [{
                "id": "followup",
                "title": "Follow-up 补充检索",
                "queries": state["queries"],
                "depends_on": [],
            }]
            parallel_groups = [["followup"]]

        if not subtasks:
            print("[execute_subtasks] 无子任务，跳过。")
            return {
                "messages": [AIMessage(content="[execute] 无子任务。")],
                "memory": memory.to_dict(),
            }

        subtask_map = {st["id"]: st for st in subtasks}

        # 如果没有 parallel_groups，所有子任务放一组并行
        if not parallel_groups:
            parallel_groups = [[st["id"] for st in subtasks]]

        print(f"[execute_subtasks] subtasks={len(subtasks)}, groups={len(parallel_groups)}")
        for gi, group in enumerate(parallel_groups):
            print(f"  group_{gi}: {group}")

        all_new_docs: List[Document] = []
        all_used_queries: List[str] = []

        # ── 按层执行 ──
        for gi, group in enumerate(parallel_groups):
            print(f"\n══ 执行并行组 {gi} / {len(parallel_groups)-1}: {group} ══")

            # 构建当前组中每个子任务的上下文（前序 findings）
            async def _run_subtask(st_id: str) -> Tuple[str, SubtaskResult]:
                st = subtask_map.get(st_id)
                if not st:
                    return st_id, SubtaskResult(
                        subtask_id=st_id,
                        title="未知",
                        success=False,
                        error="子任务未找到",
                    )

                # 如果已有 findings，跳过
                if subtask_findings.get(st_id):
                    print(f"  [{st_id}] 已有 findings，跳过。")
                    existing = subtask_findings[st_id]
                    if isinstance(existing, dict):
                        return st_id, SubtaskResult.from_dict(existing)
                    return st_id, SubtaskResult(
                        subtask_id=st_id,
                        title=st.get("title", ""),
                        findings=str(existing),
                    )

                # 汇总前序依赖的 findings
                deps = st.get("depends_on", [])
                deps_text = ""
                if deps:
                    dep_parts = []
                    for dep_id in deps:
                        if dep_id in subtask_findings:
                            dep_findings = _get_findings_text(subtask_findings[dep_id])
                            if dep_findings:
                                dep_parts.append(f"[{dep_id}] {dep_findings}")
                    if dep_parts:
                        deps_text = "\n".join(dep_parts)

                return await _process_one_subtask(
                    subtask=st,
                    question=question,
                    key_entities=key_entities,
                    query_history=query_history,
                    deps_findings=deps_text,
                    flash_llm=_flash,
                    searcher=searcher,
                    fetcher=fetcher,
                    existing_urls=existing_urls,
                    memory=memory,  # 传递 memory
                )

            # 组内并行
            results = await asyncio.gather(*[_run_subtask(st_id) for st_id in group])

            for st_id, result in results:
                # 存储 SubtaskResult.to_dict()
                subtask_findings[st_id] = result.to_dict()

                # 收集新文档
                if result.success and result.docs_fetched_count > 0:
                    all_used_queries.extend(result.queries_final)
                    # 更新 existing_urls 供后续组使用
                    existing_urls.update(result.docs_fetched_urls)

        # ── 更新 query_history ──
        query_history.extend(all_used_queries)
        dedup_history = list(dict.fromkeys(q for q in query_history if q))

        # ── 打印汇总和统计 ──
        print(f"\n[execute_subtasks] 完成！")
        print(f"[execute_subtasks] subtask_findings:")
        for st_id, result_dict in subtask_findings.items():
            findings_preview = result_dict.get("findings", "")[:100]
            duration = result_dict.get("duration_ms", 0)
            llm_calls = result_dict.get("llm_calls", 0)
            print(f"  [{st_id}] {findings_preview}... (duration={duration:.0f}ms, llm_calls={llm_calls})")

        # 打印 memory 统计
        stats = memory.get_statistics()
        print(f"[execute_subtasks] memory stats: {stats['total_steps']} steps, "
              f"success_rate={stats['success_rate']*100:.0f}%, "
              f"total_tool_calls={stats['total_tool_calls']}")

        msg = AIMessage(
            content=(
                f"[execute] 执行 {len(subtasks)} 个子任务，"
                f"全部完成。"
            )
        )

        return {
            "subtask_findings": subtask_findings,
            "query_history": dedup_history,
            "memory": memory.to_dict(),  # 保存 memory
            "messages": [msg],
        }

    return execute_subtasks
