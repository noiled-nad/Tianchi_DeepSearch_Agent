"""deepresearch/nodes/execute_subtasks.py

OAgents 风格子任务执行器：
按 parallel_groups 分层并行执行子任务，每个子任务内部走：
  rewriter (用前序 candidates 锚点重写查询)
  → query_optimize (一次 LLM 调用同时 reflect + expand)
  → search
  → fetch
  → extract_structured (提取结构化输出: sub_q, evidence, candidates)
  → self_check (自查反思，不达标则重试)

核心改进：
1. 结构化输出：每个 subtask 输出 (sub_q, evidence, candidates)
2. Rewriter：用前序 candidates 作为锚点替换泛化描述
3. Self-check：每步完成后自查，不达标则用 refined_queries 重试，同时剔除不合格候选
4. 并行执行：同组无依赖 subtask 通过 asyncio.gather 并发执行
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Set, Tuple

from langchain_core.messages import AIMessage

from ..schemas import Document, SearchResult
from ..prompt_loader import load_prompt
from ..state import DeepResearchState
from .query_optimize import _reflect_and_expand
from ..tools.compress import compress_doc
from ..goal_type import analyze_goal_type, get_extraction_instructions, get_goal_type_name


# ───────── Prompts ─────────

STRUCTURED_EXTRACTION_PROMPT = load_prompt("execute_subtasks.yaml", "structured_extraction_prompt")
REWRITER_PROMPT = load_prompt("execute_subtasks.yaml", "rewriter_prompt")
SELF_CHECK_PROMPT = load_prompt("execute_subtasks.yaml", "self_check_prompt")
INITIAL_QUERY_GEN_PROMPT = load_prompt("execute_subtasks.yaml", "initial_query_gen_prompt")


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


def _safe_json_obj(text: str) -> Dict:
    """从 LLM 输出中提取 JSON 对象。"""
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


def _build_deps_context(subtask: Dict, prev_results: Dict[str, Dict]) -> str:
    """构建依赖上下文字符串（用于传给提取函数），包含候选答案。"""
    deps = subtask.get("depends_on", [])
    if not deps or not prev_results:
        return ""
    parts = []
    for dep_id in deps:
        if dep_id in prev_results:
            r = prev_results[dep_id]
            if isinstance(r, dict):
                candidates = r.get('candidates', [])
                answer_str = candidates[0] if candidates else '(未知)'
                if len(candidates) > 1:
                    answer_str += f" （其他候选: {', '.join(candidates[1:])}）"
                parts.append(f"[{dep_id}] {r.get('sub_query', '')}: {answer_str}")
            else:
                parts.append(f"[{dep_id}] {str(r)[:200]}")
    if parts:
        return "\n前序子任务结果：\n" + "\n".join(parts)
    return ""


# ───────── 子任务内部流程 ─────────

def _build_seed_queries(
    subtask: Dict,
    prev_results: Dict[str, Dict],
) -> List[str]:
    """
    构造 seed queries（纯机械拼接，不经 LLM）：
    - 用 subtask title 的原文作为第一条 seed
    - 如果有前序依赖的 candidates，用候选答案替换 title 中的 ID 或直接拼接
    保证原文措辞零篡改，直接进搜索引擎。
    """
    seeds: List[str] = []
    title = subtask.get("title", "").strip()
    if not title:
        return seeds

    # seed 1: 优先做依赖替换；若无法替换则回退到 title 原文（去掉标点，空格分隔）
    clean_title = re.sub(r'[，。？！、；：""''（）\[\]【】]', ' ', title).strip()
    clean_title = re.sub(r'\s+', ' ', clean_title)
    if not clean_title:
        return seeds

    deps = subtask.get("depends_on", [])
    _invalid_answers = {"未找到相关文档。", "提取失败。", "未执行。", "未生成有效查询。"}

    # 先尝试用依赖任务最佳候选(candidates[0])替换占位符，作为 seed_1
    seed_1 = clean_title
    replaced_any = False
    if deps and prev_results:
        for dep_id in deps:
            r = prev_results.get(dep_id)
            if not isinstance(r, dict):
                continue
            candidates = r.get("candidates", [])
            best = str(candidates[0]).strip() if candidates else ""
            if not best or best in _invalid_answers:
                continue
            pattern = rf'\[?{dep_id}\]?|[（\(]?{dep_id}[）\)]?'
            if re.search(pattern, seed_1, re.IGNORECASE):
                seed_1 = re.sub(pattern, best, seed_1, flags=re.IGNORECASE)
                replaced_any = True

    seeds.append(seed_1.strip() if replaced_any else clean_title)

    # seed 2: 如果有前序依赖，用 candidates 替换 title 里的 ID 或拼接
    if deps and prev_results:
        for dep_id in deps:
            r = prev_results.get(dep_id)
            if isinstance(r, dict):
                # 收集所有候选答案
                candidates = r.get("candidates", [])
                if not candidates:
                    continue
                # 为每个候选生成 seed
                for cand in candidates[:3]:  # 最多取前3个候选
                    cand = str(cand).strip()
                    if not cand or cand in _invalid_answers:
                        continue
                    
                    # 尝试替换标题中引用的 ID (如 ST3, [ST3])
                    pattern = rf'\[?{dep_id}\]?|[（\(]?{dep_id}[）\)]?'
                    if re.search(pattern, clean_title, re.IGNORECASE):
                        replaced = re.sub(pattern, cand, clean_title, flags=re.IGNORECASE)
                        seeds.append(replaced.strip())
                    else:
                        # 兜底：直接拼接
                        seeds.append(f"{cand} {clean_title}")
                    
                    if len(cand) > 2 and cand not in seeds:
                        seeds.append(cand)

    return seeds


async def _optimize_queries_for_subtask(
    flash_llm,
    question: str,
    subtask: Dict,
    key_entities: List[str],
    query_history: List[str],
) -> List[str]:
    """
    为单个子任务优化查询（一次 LLM 调用同时完成反思 + 扩展）。
    依赖上下文注入由上层 Rewriter 负责，此函数仅做查询质量优化。
    """
    raw_queries = subtask.get("queries", [])
    if not raw_queries:
        return []

    # 构建子任务聚焦上下文
    st_title = subtask.get("title", "")
    st_reason = subtask.get("reason", "")
    subtask_context = f"{st_title}。{st_reason}" if st_reason else st_title

    # 一次 LLM 调用：反思 + 扩展
    expanded = await _reflect_and_expand(
        flash_llm, subtask_context, raw_queries, key_entities, query_history
    )

    # 保留原始查询作为兜底，避免语义漂移
    all_queries = list(raw_queries) + expanded

    # 去重
    deduped = _dedup_queries(all_queries, {q.lower() for q in query_history})
    return deduped[:10]

async def _search_fetch_compress(
    queries: List[str],
    searcher,
    fetcher,
    flash_llm,
    question: str,
    subtask: Dict,
    existing_urls: Set[str],
    max_docs: int = 6,
    per_query_results: int = 3,
) -> Tuple[List[Document], List[str], List[str]]:
    """
    搜索 → 抓取 → 压缩 流水线。
    asyncio.as_completed 实现原子级并发：谁先抓完谁先压缩。
    网络 I/O 和 LLM I/O 完美交叠，互不阻塞。
    Returns: (raw_docs, used_queries, compressed_snippets)
    """
    seen_urls = set(existing_urls)
    candidate_items: List[Tuple[str, str]] = []  # (url, query)

    # Search
    search_errors = 0
    total_search_results = 0
    for query in queries:
        if len(candidate_items) >= max_docs:
            break
        try:
            results: List[SearchResult] = await searcher.search(query)
        except Exception as e:
            search_errors += 1
            print(f"    [search] FAILED query='{query[:60]}': {type(e).__name__}: {e}")
            continue
        total_search_results += len(results)
        if not results:
            print(f"    [search] 0 results for query='{query[:60]}'")
        for r in results[:per_query_results]:
            if len(candidate_items) >= max_docs:
                break
            if not r.url or r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            candidate_items.append((r.url, query))

    print(f"    [search] queries={len(queries)}, results={total_search_results}, "
          f"candidates={len(candidate_items)}, errors={search_errors}")

    if not candidate_items:
        return [], queries, []

    # ── Fetch → Compress 流水线 ──
    fetch_timeout_s = 45.0

    async def _fetch_then_compress(url: str, query: str):
        """原子级流水线：fetch → compress"""
        try:
            async def _call_fetch():
                try:
                    return await fetcher.fetch(url, query=query)
                except TypeError:
                    return await fetcher.fetch(url)
            doc = await asyncio.wait_for(_call_fetch(), timeout=fetch_timeout_s)
        except asyncio.TimeoutError:
            print(f"    [fetch] TIMEOUT({fetch_timeout_s}s) url='{url[:80]}'")
            return None, None
        except Exception as e:
            print(f"    [fetch] FAILED url='{url[:80]}': {type(e).__name__}: {e}")
            return None, None

        if doc is None:
            return None, None

        # 紧跟 fetch，立刻压缩
        snippet = await compress_doc(flash_llm, question, subtask, doc)
        return doc, snippet

    raw_docs: List[Document] = []
    snippets: List[str] = []
    failures = 0
    tasks = [asyncio.create_task(_fetch_then_compress(u, q)) for u, q in candidate_items]
    for fut in asyncio.as_completed(tasks):
        doc, snippet = await fut
        if doc is not None:
            raw_docs.append(doc)
            if snippet:
                snippets.append(snippet)
        else:
            failures += 1
        if len(raw_docs) >= max_docs:
            break

    # 取消剩余未完成任务
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if failures > 0:
        print(f"    [pipeline] {failures}/{len(candidate_items)} failed")
    print(f"    [pipeline] docs={len(raw_docs)}, snippets={len(snippets)}")

    return raw_docs, queries, snippets


async def _rewrite_queries_with_context(
    flash_llm,
    question: str,
    subtask: Dict,
    prev_results: Dict[str, Dict],
) -> List[str]:
    """
    Rewriter：为子任务生成/重写搜索查询。
    - 无前序结果时：根据 subtask title/reason + question 生成初始查询
    - 有前序结果时：用 candidates 作为锚点重写查询
    """
    deps = subtask.get("depends_on", [])
    raw_queries = subtask.get("queries", [])

    # 收集前序结构化结果（无论是否有依赖，都检查所有已完成结果）
    relevant_prev = {}
    if deps and prev_results:
        for dep_id in deps:
            if dep_id in prev_results:
                relevant_prev[dep_id] = prev_results[dep_id]

    # ── 情况1: 有前序依赖结果 → 用 candidates 锚点重写 ──
    if relevant_prev:
        prev_parts = []
        # 预处理 title 和 reason，替换其中的 [STx] 占位符
        st_title = subtask.get("title", "")
        st_reason = subtask.get("reason", "")
        
        for dep_id, r in relevant_prev.items():
            if isinstance(r, dict):
                candidates = r.get("candidates", [])
                best = candidates[0] if candidates else ""
                
                # 替换标题和原因中的 ID
                pattern = rf'\[?{dep_id}\]?|[（\(]?{dep_id}[）\)]?'
                if best and best not in {"未找到相关文档。", "提取失败。", "未执行。"}:
                    st_title = re.sub(pattern, best, st_title, flags=re.IGNORECASE)
                    st_reason = re.sub(pattern, best, st_reason, flags=re.IGNORECASE)

                candidates_str = ", ".join(candidates) if candidates else "(未知)"
                prev_parts.append(
                    f"[{dep_id}] 问题: {r.get('sub_query', '')}\n"
                    f"  候选答案: {candidates_str}\n"
                    f"  证据: {'; '.join(r.get('evidence', [])[:3])}"
                )
            else:
                prev_parts.append(f"[{dep_id}] {str(r)[:200]}")

        prompt = REWRITER_PROMPT.format(
            question=question,
            subtask_id=subtask["id"],
            subtask_title=st_title, # 使用替换后的 title
            subtask_reason=st_reason, # 使用替换后的 reason
            original_queries="\n".join(f"- {q}" for q in raw_queries) if raw_queries else "（无原始查询，请根据前序结果和子任务目标生成）",
            prev_results="\n".join(prev_parts),
        )

        try:
            resp = await flash_llm.ainvoke(prompt)
            refined = [ln.strip().lstrip("- ·•0123456789.)") for ln in str(resp.content).split("\n") if ln.strip()]
            refined = [q for q in refined if len(q) > 3]
            if refined:
                print(f"  [{subtask['id']}] rewriter: {len(refined)} queries anchored by candidates")
                return refined + raw_queries
        except Exception as e:
            print(f"  [{subtask['id']}] rewriter failed: {e}")

        return raw_queries if raw_queries else [subtask.get("title", question[:60])]

    # ── 情况2: 无前序结果 → 生成初始查询 ──
    if not raw_queries:
        prompt = INITIAL_QUERY_GEN_PROMPT.format(
            question=question,
            subtask_id=subtask["id"],
            subtask_title=subtask.get("title", ""),
            subtask_reason=subtask.get("reason", ""),
        )
        try:
            resp = await flash_llm.ainvoke(prompt)
            generated = [ln.strip().lstrip("- ·•0123456789.)") for ln in str(resp.content).split("\n") if ln.strip()]
            generated = [q for q in generated if len(q) > 3]
            if generated:
                print(f"  [{subtask['id']}] initial_query_gen: {len(generated)} queries generated")
                return generated
        except Exception as e:
            print(f"  [{subtask['id']}] initial_query_gen failed: {e}")
        # 兜底：用 title 作为查询
        return [subtask.get("title", question[:60])]

    return raw_queries


async def _extract_structured_findings(
    flash_llm,
    question: str,
    subtask: Dict,
    snippets: List[str],
    deps_context: str = "",
) -> Dict:
    """
    用 LLM 从压缩片段中提取结构化发现：{sub_query, evidence, candidates, confidence, sources}。
    """
    if not snippets:
        return {
            "sub_query": subtask.get("title", ""),
            "evidence": [],
            "candidates": [],
            "confidence": 0.0,
            "sources": [],
        }

    # ── Goal Type 分析 ──
    subtask_title = subtask.get("title", "")
    subtask_reason = subtask.get("reason", "")
    goal_type = analyze_goal_type(subtask_title, subtask_reason, question)
    goal_type_name = get_goal_type_name(goal_type)
    extraction_hint = get_extraction_instructions(goal_type)
    print(f"  [{subtask["id"]}] goal_type={goal_type_name}")

    docs_text = "\n\n".join(f"[D{i}] {s}" for i, s in enumerate(snippets, 1))
    prompt = STRUCTURED_EXTRACTION_PROMPT.format(
        question=question,
        subtask_id=subtask["id"],
        subtask_title=subtask.get("title", ""),
        subtask_reason=subtask.get("reason", ""),
        deps_context=deps_context,
        docs_text=docs_text,
        goal_type_hint=f"\n【提取策略：{goal_type_name}】\n{extraction_hint}\n",
    )

    try:
        resp = await flash_llm.ainvoke(prompt)
        raw = str(resp.content).strip()
        obj = _safe_json_obj(raw)
        candidates = [str(c).strip() for c in obj.get("candidates", []) if str(c).strip()]
        # 兼容：如果 LLM 还是输出了 sub_answer，将其插入 candidates 开头
        legacy_answer = str(obj.get("sub_answer", "")).strip()
        if legacy_answer and legacy_answer not in candidates:
            candidates.insert(0, legacy_answer)
        return {
            "sub_query": str(obj.get("sub_query", subtask.get("title", ""))).strip(),
            "evidence": [str(e).strip() for e in obj.get("evidence", []) if str(e).strip()],
            "candidates": candidates,
            "confidence": min(1.0, max(0.0, float(obj.get("confidence", 0.0)))),
            "sources": [str(s).strip() for s in obj.get("sources", []) if str(s).strip()],
        }
    except Exception as e:
        print(f"  [{subtask['id']}] extract_structured failed: {e}")
        return {
            "sub_query": subtask.get("title", ""),
            "evidence": [],
            "candidates": [],
            "confidence": 0.0,
            "sources": [],
        }


async def _self_check_subtask(
    flash_llm,
    question: str,
    subtask: Dict,
    result: Dict,
) -> tuple:
    """
    自查子任务结果是否达标。
    会从 result 中原地剔除被拒绝的 candidates。
    Returns: (passed: bool, refined_queries: List[str])
    """
    evidence_text = "\n".join(f"- {e}" for e in result.get("evidence", [])) or "（无证据）"
    candidates = result.get("candidates", [])
    candidates_text = ", ".join(candidates) if candidates else "（无候选）"
    prompt = SELF_CHECK_PROMPT.format(
        question=question,
        subtask_id=subtask["id"],
        subtask_title=subtask.get("title", ""),
        subtask_reason=subtask.get("reason", ""),
        sub_query=result.get("sub_query", ""),
        evidence=evidence_text,
        candidates=candidates_text,
        confidence=result.get("confidence", 0),
    )

    try:
        resp = await flash_llm.ainvoke(prompt)
        raw = str(resp.content).strip()
        obj = _safe_json_obj(raw)
        passed = bool(obj.get("passed", True))
        refined_queries = [str(q).strip() for q in obj.get("refined_queries", []) if str(q).strip()]
        rejected = [str(r).strip() for r in obj.get("rejected_candidates", []) if str(r).strip()]
        reason = str(obj.get("reason", "")).strip()

        # ── 从 candidates 中剔除被拒绝的 ──
        if rejected and candidates:
            rejected_set = set(rejected)
            before_count = len(candidates)
            pruned = [c for c in candidates if c not in rejected_set]
            if pruned:
                result["candidates"] = pruned
                print(f"  [{subtask['id']}] self_check: 剔除 {before_count - len(pruned)} 个候选 {rejected}，剩余 {len(pruned)} 个")
                # 剔除后还有候选 → 视为通过
                passed = True
            else:
                # 全部被剔除 → 不通过
                print(f"  [{subtask['id']}] self_check: 所有候选均被剔除")
                passed = False

        if not passed:
            print(f"  [{subtask['id']}] self_check FAILED: {reason}")
            if refined_queries:
                print(f"  [{subtask['id']}] refined_queries: {refined_queries}")
        else:
            print(f"  [{subtask['id']}] self_check PASSED: {reason}")

        return passed, refined_queries
    except Exception as e:
        print(f"  [{subtask['id']}] self_check error: {e}")
        return True, []  # 出错时默认通过，避免死循环


async def _process_one_subtask(
    subtask: Dict,
    question: str,
    key_entities: List[str],
    query_history: List[str],
    prev_results: Dict[str, Dict],
    flash_llm,
    searcher,
    fetcher,
    existing_urls: Set[str],
    max_retries: int = 1,
) -> Tuple[str, List[Document], List[str], Dict]:
    """
    处理单个子任务完整流程（含 Rewriter + 自查重试）：
    rewrite → optimize → search → fetch → extract_structured → self_check → (retry)
    返回 (subtask_id, new_docs, used_queries, structured_result)。
    """
    st_id = subtask["id"]
    print(f"\n  ── 执行子任务 [{st_id}] {subtask.get('title', '')} ──")

    all_new_docs: List[Document] = []
    all_used_queries: List[str] = []
    all_snippets: List[str] = []
    retry_queries: List[str] = []
    result: Dict = {
        "sub_query": subtask.get("title", ""),
        "evidence": [],
        "candidates": [],
        "confidence": 0.0,
        "sources": [],
    }

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"  [{st_id}] === 自查重试 第{attempt}次 ===")

        # ── Phase A: Seed（search → fetch → compress 流水线） ──
        seed_queries = _build_seed_queries(subtask, prev_results)
        if attempt > 0 and retry_queries:
            seed_queries = retry_queries + seed_queries
        seed_queries = _dedup_queries(seed_queries, {q.lower() for q in query_history})
        print(f"  [{st_id}] seed_queries={len(seed_queries)}")
        for i, q in enumerate(seed_queries, 1):
            print(f"  [{st_id}]   seed_{i}: {q}")

        seed_docs, seed_used, seed_snippets = await _search_fetch_compress(
            seed_queries, searcher, fetcher, flash_llm, question, subtask,
            existing_urls, max_docs=3,
        )
        for d in seed_docs:
            if getattr(d, "url", None):
                existing_urls.add(d.url)

        # ── Phase B: LLM 查询 → optimize → search → fetch → compress ──
        rewritten = await _rewrite_queries_with_context(
            flash_llm, question, subtask, prev_results
        )
        # 解析 P/R 标签（如果有的话），去掉标签前缀
        cleaned_rewritten = []
        for q in rewritten:
            q_clean = re.sub(r'^[PR][：:]\s*', '', q).strip()
            if q_clean:
                cleaned_rewritten.append(q_clean)
        rewritten = cleaned_rewritten if cleaned_rewritten else rewritten
        print(f"  [{st_id}] llm_queries={len(rewritten)}")

        # Optimize (reflect + rollout)
        temp_subtask = {**subtask, "queries": rewritten}
        optimized = await _optimize_queries_for_subtask(
            flash_llm, question, temp_subtask, key_entities, query_history
        )
        print(f"  [{st_id}] optimized_queries={len(optimized)}")
        for i, q in enumerate(optimized, 1):
            print(f"  [{st_id}]   opt_{i}: {q}")

        if optimized:
            opt_docs, opt_used, opt_snippets = await _search_fetch_compress(
                optimized, searcher, fetcher, flash_llm, question, subtask,
                existing_urls, max_docs=4,
            )
        else:
            opt_docs, opt_used, opt_snippets = [], [], []
        for d in opt_docs:
            if getattr(d, "url", None):
                existing_urls.add(d.url)

        # ── 合并 ──
        new_docs = seed_docs + opt_docs
        used_qs = seed_used + opt_used
        all_new_docs.extend(new_docs)
        all_used_queries.extend(used_qs)
        all_snippets.extend(seed_snippets + opt_snippets)
        print(f"  [{st_id}] docs={len(new_docs)}, snippets={len(seed_snippets)+len(opt_snippets)}")

        # 4) 从压缩片段提取结构化发现
        deps_context = _build_deps_context(subtask, prev_results)
        result = await _extract_structured_findings(
            flash_llm, question, subtask, all_snippets, deps_context
        )
        print(f"  [{st_id}] candidates: {result.get('candidates', [])}")
        print(f"  [{st_id}] confidence: {result.get('confidence', 0)}")

        # 5) 自查：如果未达标且还有重试机会，则重试
        if attempt < max_retries:
            passed, retry_queries = await _self_check_subtask(
                flash_llm, question, subtask, result
            )
            if passed:
                break
            # 未通过 → 下一轮重试
        # 最后一次尝试 → 接受结果

    return st_id, all_new_docs, all_used_queries, result


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
        subtask_findings: Dict[str, Any] = dict(state.get("subtask_findings", {}) or {})

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

            # 构建子任务执行闭包
            async def _run_subtask(st_id: str):
                st = subtask_map.get(st_id)
                if not st:
                    return st_id, [], [], {
                        "sub_query": "", "evidence": [],
                        "candidates": [], 
                        "confidence": 0.0, "sources": [],
                    }

                # 如果已有结果，跳过
                if subtask_findings.get(st_id):
                    print(f"  [{st_id}] 已有结果，跳过。")
                    return st_id, [], [], subtask_findings[st_id]

                return await _process_one_subtask(
                    subtask=st,
                    question=question,
                    key_entities=key_entities,
                    query_history=query_history,
                    prev_results=subtask_findings,
                    flash_llm=_flash,
                    searcher=searcher,
                    fetcher=fetcher,
                    existing_urls=existing_urls,
                )

            # 组内并行
            results = await asyncio.gather(*[_run_subtask(st_id) for st_id in group])

            for st_id, new_docs, used_qs, result in results:
                subtask_findings[st_id] = result
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
        for st_id, result in subtask_findings.items():
            if isinstance(result, dict):
                cands = result.get('candidates', [])
                best = cands[0] if cands else '(无答案)'
                print(f"  [{st_id}] best={best[:80]} cands={cands} conf={result.get('confidence', 0)}")
            else:
                print(f"  [{st_id}] {str(result)[:100]}...")

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
