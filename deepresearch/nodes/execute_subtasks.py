"""deepresearch/nodes/execute_subtasks.py

OAgents 风格子任务执行器：
按 parallel_groups 分层并行执行子任务，每个子任务内部走：
  query_optimize (batch reflect + parallel rollout)
  → search
  → fetch
  → extract_findings (LLM 抽取关键发现)

后序子任务会注入前序子任务的 findings 作为上下文，
实现级联推理（多跳问题的核心价值）。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, List, Set, Tuple

from langchain_core.messages import AIMessage

from ..schemas import Document, SearchResult
from ..prompt_loader import load_prompt
from ..state import DeepResearchState
from .query_optimize import _reflect_batch, _rollout_query


# ───────── Prompts ─────────

FINDING_EXTRACTION_PROMPT = load_prompt("execute_subtasks.yaml", "finding_extraction_prompt")
DEPS_QUERY_REFINE_PROMPT = load_prompt("execute_subtasks.yaml", "deps_query_refine_prompt")


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


# ───────── 子任务内部流程 ─────────

async def _optimize_queries_for_subtask(
    flash_llm,
    question: str,
    subtask: Dict,
    key_entities: List[str],
    query_history: List[str],
    deps_findings: str = "",
) -> List[str]:
    """
    为单个子任务优化查询：
    1) 如果有前序 findings，先用 LLM 基于 findings 细化查询
    2) batch reflect
    3) parallel rollout
    4) 去重
    """
    raw_queries = subtask.get("queries", [])
    if not raw_queries:
        return []

    # ── 构建子任务聚焦上下文（不传完整问题，避免 reflect/rollout 跑偏到其他子任务） ──
    st_title = subtask.get("title", "")
    st_reason = subtask.get("reason", "")
    subtask_context = f"{st_title}。{st_reason}" if st_reason else st_title

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
                # 合并：细化查询 + 原始查询
                raw_queries = refined + raw_queries
                print(f"  [{subtask['id']}] deps_refine: +{len(refined)} queries from findings")
        except Exception as e:
            print(f"  [{subtask['id']}] deps_refine failed: {e}")

    # Batch reflect（传子任务上下文而非完整问题）
    reflect_results = await _reflect_batch(flash_llm, subtask_context, raw_queries)
    reflected = [aug for (_, aug) in reflect_results]

    # Parallel rollout（同样传子任务上下文）
    rollout_n = max(2, min(3, 8 // max(1, len(reflected))))

    async def _do_rollout(q: str) -> List[str]:
        return await _rollout_query(flash_llm, subtask_context, q, key_entities, query_history, rollout_n)

    rollout_results = await asyncio.gather(*[_do_rollout(q) for q in reflected])

    # 重要：保留原始查询，避免优化阶段发生语义漂移（如“近距离作战武器”被误改成“近战武器”）
    all_queries = list(raw_queries) + list(reflected)
    for variants in rollout_results:
        all_queries.extend(variants)

    # 去重
    deduped = _dedup_queries(all_queries, {q.lower() for q in query_history})
    return deduped[:10]  # 每个子任务最多 10 条优化查询


async def _search_and_fetch(
    queries: List[str],
    searcher,
    fetcher,
    existing_urls: Set[str],
    max_docs: int = 6,
    per_query_results: int = 3,
    parallelism: int = 3,
) -> Tuple[List[Document], List[str]]:
    """
    搜索 + 抓取，返回 (new_docs, searched_queries)。
    复用 retrieve.py 的核心逻辑，但无需 state。
    """
    seen_urls = set(existing_urls)
    candidate_items: List[Tuple[str, str]] = []  # (url, query)

    # Search
    for query in queries:
        if len(candidate_items) >= max_docs:
            break
        try:
            results: List[SearchResult] = await searcher.search(query)
        except Exception:
            continue
        for r in results[:per_query_results]:
            if len(candidate_items) >= max_docs:
                break
            if not r.url or r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            candidate_items.append((r.url, query))

    if not candidate_items:
        return [], queries

    # Fetch（并发）
    sem = asyncio.Semaphore(parallelism)

    async def _fetch_one(url: str, query: str):
        async with sem:
            try:
                try:
                    return await fetcher.fetch(url, query=query)
                except TypeError:
                    return await fetcher.fetch(url)
            except Exception:
                return None

    new_docs: List[Document] = []
    tasks = [asyncio.create_task(_fetch_one(u, q)) for (u, q) in candidate_items]
    for fut in asyncio.as_completed(tasks):
        doc = await fut
        if doc is not None:
            new_docs.append(doc)
        if len(new_docs) >= max_docs:
            break

    return new_docs, queries


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
) -> Tuple[str, List[Document], List[str], str]:
    """
    处理单个子任务完整流程：optimize → search → fetch → extract。
    返回 (subtask_id, new_docs, used_queries, findings_text)。
    """
    st_id = subtask["id"]
    print(f"\n  ── 执行子任务 [{st_id}] {subtask.get('title', '')} ──")

    # 1) 优化查询
    deps_context = ""
    if deps_findings:
        deps_context = f"\n前序子任务发现：\n{deps_findings}"

    optimized = await _optimize_queries_for_subtask(
        flash_llm, question, subtask, key_entities, query_history, deps_findings
    )
    print(f"  [{st_id}] optimized_queries={len(optimized)}")
    for i, q in enumerate(optimized, 1):
        print(f"  [{st_id}]   q{i}: {q}")

    if not optimized:
        return st_id, [], [], "未生成有效查询。"

    # 2) 搜索 + 抓取
    new_docs, used_queries = await _search_and_fetch(
        optimized, searcher, fetcher, existing_urls
    )
    print(f"  [{st_id}] fetched_docs={len(new_docs)}")

    # 3) 抽取发现
    findings = await _extract_findings(
        flash_llm, question, subtask, new_docs, deps_context
    )
    print(f"  [{st_id}] findings: {findings[:120]}...")

    return st_id, new_docs, used_queries, findings


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
        subtask_findings: Dict[str, str] = dict(state.get("subtask_findings", {}) or {})

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
            async def _run_subtask(st_id: str):
                st = subtask_map.get(st_id)
                if not st:
                    return st_id, [], [], "子任务未找到。"

                # 如果已有 findings，跳过
                if subtask_findings.get(st_id):
                    print(f"  [{st_id}] 已有 findings，跳过。")
                    return st_id, [], [], subtask_findings[st_id]

                # 汇总前序依赖的 findings
                deps = st.get("depends_on", [])
                deps_text = ""
                if deps:
                    dep_parts = []
                    for dep_id in deps:
                        if dep_id in subtask_findings:
                            dep_parts.append(f"[{dep_id}] {subtask_findings[dep_id]}")
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
                )

            # 组内并行
            results = await asyncio.gather(*[_run_subtask(st_id) for st_id in group])

            for st_id, new_docs, used_qs, findings in results:
                subtask_findings[st_id] = findings
                all_new_docs.extend(new_docs)
                all_used_queries.extend(used_qs)
                # 更新 existing_urls 供后续组使用
                for d in new_docs:
                    if getattr(d, "url", None):
                        existing_urls.add(d.url)

        # ── 更新 query_history ──
        query_history.extend(all_used_queries)
        dedup_history = list(dict.fromkeys(q for q in query_history if q))

        merged_docs = existing_docs + all_new_docs

        # 打印汇总
        print(f"\n[execute_subtasks] 完成！new_docs={len(all_new_docs)}, total_docs={len(merged_docs)}")
        print(f"[execute_subtasks] subtask_findings:")
        for st_id, findings in subtask_findings.items():
            print(f"  [{st_id}] {findings[:100]}...")

        msg = AIMessage(
            content=(
                f"[execute] 执行 {len(subtasks)} 个子任务，"
                f"新增文档 {len(all_new_docs)}，累计 {len(merged_docs)}。"
            )
        )

        return {
            "documents": merged_docs,
            "subtask_findings": subtask_findings,
            "query_history": dedup_history,
            "messages": [msg],
        }

    return execute_subtasks
