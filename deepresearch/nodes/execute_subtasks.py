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
from ..memory import ExecutionMemory, StepRecord, generate_group_summary
from ..context_manager import build_task_packet, build_candidate_packet


# ───────── Prompts ─────────

STRUCTURED_EXTRACTION_PROMPT = load_prompt("execute_subtasks.yaml", "structured_extraction_prompt")
VERIFICATION_QUERY_GEN_PROMPT = load_prompt("execute_subtasks.yaml", "verification_query_gen_prompt")
VERIFY_AND_FILTER_PROMPT = load_prompt("execute_subtasks.yaml", "verify_and_filter_prompt")
FAILURE_REFLECT_PROMPT = load_prompt("execute_subtasks.yaml", "failure_reflect_prompt")
INITIAL_QUERY_GEN_PROMPT = load_prompt("execute_subtasks.yaml", "initial_query_gen_prompt")



# ───────── 题型专属 Query 模板 ─────────

PROBLEM_TYPE_TEMPLATES: Dict[str, str] = {
    "entity_chain": (
        "### entity_chain（多跳实体链）\n"
        "- A（Anchor）：单独用【已知最独特的实体全名】搜索\n"
        "- E（Evidence）：【确切个体名】+【空格】+【目标属性/职位/直接关系名词】\n"
        "- V（Verify）：【待验证候选】+【空格】+【约束条件(如年份/身份)】\n"
        "\n"
        "示例：\n"
        "  A: {具体的机构全名或人名} 创始人\n"
        "  E: {前序查出的孤立实体} {属性名词}\n"
        "  V: {候选孤立实体} {附加约束条件}"
    ),
    "document_lookup": (
        "### document_lookup（文档/论文/定位）\n"
        "- A（Anchor）：【核心论文/文档名】+ pdf/archive/site\n"
        "- E（Evidence）：【文档名】+【空格】+【目标属性如:作者/出处/致谢】\n"
        "- V（Verify）：【文档名】+【约束年份/机构名】\n"
        "\n"
        "示例：\n"
        "  A: \"{具体的论文全称/简称}\" arxiv\n"
        "  E: {前序查出的确切文档名} 章节名词\n"
        "  V: {候选文档名} {核验关键词}"
    ),
    "year_resolution": (
        "### year_resolution（年份查找）\n"
        "- A（Anchor）：【已知核心短句/特定实体】+ 年份\n"
        "- E（Evidence）：【前序推导出的确切年份】+【相关事件核心词】\n"
        "- V（Verify）：【年份候选数字】+【前后置附加约束】\n"
        "\n"
        "示例：\n"
        "  A: {专有事件名} 时间\n"
        "  E: {确认前序的年份数字名} {其他事件关联词}\n"
        "  V: {候选年份} {核验词}"
    ),
    "work_identification": (
        "### work_identification（作品/角色识别）\n"
        "- A（Anchor）：【独特的角色名或单句台词】+ 电影/游戏等载体分类\n"
        "- E（Evidence）：【前序定死的作品名】+ 导演/制作/改编词\n"
        "- V（Verify）：【候选单一作品名】+【附加的时间或平台词】\n"
        "\n"
        "示例：\n"
        "  A: {独一无二角色名} 电视剧\n"
        "  E: {前序找到的确切作品名} 导演\n"
        "  V: {候选独立作品名} {年份等核验}"
    ),
    "science_chain": (
        "### science_chain（科学机制）\n"
        "- A（Anchor）：【核心名词:疾病/药物/分子】+ mechanism/pathway\n"
        "- E（Evidence）：【前序特定生化词】+ 表达/直接下游\n"
        "- V（Verify）：【候选单独蛋白名】+ inhibitor/副作用\n"
        "\n"
        "示例：\n"
        "  A: {孤立蛋白名} pathway\n"
        "  E: {前序具体学名} downstream\n"
        "  V: {候选学名} {临床约束}"
    ),
    "rule_check": (
        "### rule_check（规则/条款）\n"
        "- A（Anchor）：【精确法案名/规定名缩写】+ 条文\n"
        "- E（Evidence）：【特定法案名】+【对象属性/条件名词】\n"
        "- V（Verify）：【具体条款编号】+ 例外\n"
        "\n"
        "示例：\n"
        "  A: {具体精确法案缩写} 法规\n"
        "  E: {特定法案词} 适用范围\n"
        "  V: {条款编号} 豁免条件"
    ),
    "field_extraction": (
        "### field_extraction（字段/数据）\n"
        "- A（Anchor）：【具体统计机构/表单名】+ 年报/数据库\n"
        "- E（Evidence）：【前序定位的确切实体/文档名】+【统计字段名】\n"
        "- V（Verify）：【单独数据候选】+【特定单位】\n"
        "\n"
        "示例：\n"
        "  A: {来源孤立实体名} 年表\n"
        "  E: {确切统计表名} {目标数值指标名}\n"
        "  V: {候选数字} {单位要求}"
    ),
}

# 通用兜底模板（当 problem_type 不在字典中时使用）
_DEFAULT_TYPE_TEMPLATE = (
    "### 通用模板\n"
    "- A（Anchor）：【最核心的具体孤立实体】+【空格】+【指代特征】\n"
    "- E（Evidence）：【已知确切的一跳实体名】+【空格】+【目标关系名词】\n"
    "- V（Verify）：【待验证且单独的答案词】+【核验约束词】\n"
    "\n"
    "示例：\n"
    "  A: {最独一无二实体名词} {特征形容词汇}\n"
    "  E: {前序拿到确切回答名词} {所需关系名词}\n"
    "  V: {提取的一个备选答案词} {核验证交词汇}"
)


def _get_type_template(problem_type: str) -> str:
    """根据 problem_type 获取对应的 query 设计模板。
    
    支持 primary + secondary 格式（如 "entity_chain + year_resolution"）：
    会拼接两个模板。
    """
    if not problem_type:
        return _DEFAULT_TYPE_TEMPLATE

    # 解析可能的 primary + secondary 格式
    parts = [p.strip().lower() for p in re.split(r'[+,;/]', problem_type) if p.strip()]
    
    templates = []
    for part in parts:
        # 尝试精确匹配
        if part in PROBLEM_TYPE_TEMPLATES:
            templates.append(PROBLEM_TYPE_TEMPLATES[part])
        else:
            # 尝试模糊匹配（包含关系）
            matched = False
            for key, tmpl in PROBLEM_TYPE_TEMPLATES.items():
                if part in key or key in part:
                    templates.append(tmpl)
                    matched = True
                    break
            if not matched:
                templates.append(f"### {part}（未知题型，参考通用策略）\n" + _DEFAULT_TYPE_TEMPLATE)

    if not templates:
        return _DEFAULT_TYPE_TEMPLATE

    return "\n\n".join(templates)


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


def _dedup_strings(values: List[str]) -> List[str]:
    """保持顺序去重，常用于 candidates / sources / reasons。"""
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _extend_unique_docs(target: List[Document], new_docs: List[Document], seen_urls: Set[str]) -> None:
    """按 URL 去重合并文档，避免候选分支回灌时出现重复文档。"""
    for doc in new_docs:
        url = getattr(doc, "url", None)
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        target.append(doc)


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

    # 不再强行混入 raw_queries，避免重试时受限，完全依赖 optimize 泛化出的新查询
    all_queries = expanded

    # 去重
    deduped = _dedup_queries(all_queries, {q.lower() for q in query_history})
    return deduped[:5]

async def _search_fetch_compress(
    queries: List[str],
    searcher,
    fetcher,
    flash_llm,
    question: str,
    subtask: Dict,
    existing_urls: Set[str],
    max_docs: int = 8,
    per_query_results: int = 3,
) -> Tuple[List[Document], List[str], List[str]]:
    """
    搜索 → 抓取 → 压缩 流水线。
    asyncio.as_completed 实现原子级并发：谁先抓完谁先压缩。
    网络 I/O 和 LLM I/O 完美交叠，互不阻塞。
    Returns: (raw_docs, used_queries, compressed_snippets)
    """
    seen_urls = set(existing_urls)
    candidate_items: List[Tuple[SearchResult, str]] = []  # (search_result, query)

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
            candidate_items.append((r, query))

    print(f"    [search] queries={len(queries)}, results={total_search_results}, "
          f"candidates={len(candidate_items)}, errors={search_errors}")
    for idx, (result, query) in enumerate(candidate_items[:6], 1):
        title = (result.title or "").strip().replace("\n", " ")
        print(
            f"    [search-hit-{idx}] query='{query[:50]}' "
            f"title='{title[:80]}' url='{(result.url or '')[:120]}'"
        )

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
    tasks = [asyncio.create_task(_fetch_then_compress(result.url, q)) for result, q in candidate_items]
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


async def _generate_queries(
    flash_llm,
    question: str,
    subtask: Dict,
    prev_results: Dict[str, Dict],
    brief: Dict | None = None,
    task_packet: Dict[str, Any] | None = None,
    retry_context: str = "",
) -> List[str]:
    """
    统一 query 生成入口（始终使用 INITIAL_QUERY_GEN_PROMPT）。

    无论是否有前序依赖，都走同一套 prompt：
    - dependency_context 由 _build_deps_context 构建（有依赖时带 candidates）
    - retry_context 由 self_check 失败后注入（上轮诊断 + 失败经验）
    - 题型模板由 _get_type_template 按 problem_type 选取
    """
    brief = brief or {}
    problem_type = brief.get("problem_type", "entity_chain")
    packet_constraints = list((task_packet or {}).get("constraints", []) or [])
    hard_constraints = _dedup_strings(list(brief.get("hard_constraints", []) or []) + packet_constraints)
    answer_format = brief.get("answer_format", "简洁文本答案")

    # ── 构建依赖上下文 ──
    dependency_context = (
        (task_packet or {}).get("curated_context", "")
        or _build_deps_context(subtask, prev_results)
        or "（无前序结果）"
    )

    # ── 构建 prompt ──
    type_template = _get_type_template(problem_type)

    # 包装 retry_context：有内容时加标题，无内容时留空
    formatted_retry = ""
    if retry_context:
        formatted_retry = (
            "## ⚠️ 上轮经验反馈（请务必参考，换策略重新检索）\n"
            f"{retry_context}"
        )

    prompt = INITIAL_QUERY_GEN_PROMPT.format(
        question=question,
        subtask_id=subtask["id"],
        subtask_title=subtask.get("title", ""),
        subtask_reason=subtask.get("reason", ""),
        dependency_context=dependency_context,
        hard_constraints="、".join(hard_constraints) if hard_constraints else "（无）",
        answer_format=answer_format,
        problem_type=problem_type,
        type_specific_templates=type_template,
        retry_context=formatted_retry,
    )

    try:
        resp = await flash_llm.ainvoke(prompt)
        print(resp)
        lines = [ln.strip().lstrip("- ·•0123456789.)") for ln in str(resp.content).split("\n") if ln.strip()]
        # 剥掉 A/E/V 标签
        cleaned = []
        for q in lines:
            q_clean = re.sub(r'^[AEV][：:]\s*', '', q).strip()
            if q_clean and len(q_clean) > 3:
                cleaned.append(q_clean)
        queries = cleaned if cleaned else [q for q in lines if len(q) > 3]
        max_queries = int((task_packet or {}).get("budget", {}).get("max_queries", 5))
        queries = queries[:max_queries]
        if queries:
            tag = "retry_query_gen" if retry_context else "query_gen"
            print(f"  [{subtask['id']}] {tag}({problem_type}): {len(queries)} queries")
            return queries
    except Exception as e:
        print(f"  [{subtask['id']}] query_gen failed: {e}")

    # 兜底
    return [subtask.get("title", question[:60])]


async def _extract_structured_findings(
    flash_llm,
    question: str,
    subtask: Dict,
    snippets: List[str],
    chain_memory: str = "",
) -> Dict:
    """
    用 LLM 从压缩片段中提取结构化发现：{sub_query, evidence, candidates, confidence, sources, reasoning_trace}。
    """
    if not snippets:
        return {
            "sub_query": subtask.get("title", ""),
            "evidence": [],
            "candidates": [],
            "confidence": 0.0,
            "sources": [],
            "reasoning_trace": "",
        }

    docs_text = "\n\n".join(f"[D{i}] {s}" for i, s in enumerate(snippets, 1))
    prompt = STRUCTURED_EXTRACTION_PROMPT.format(
        question=question,
        subtask_id=subtask["id"],
        subtask_title=subtask.get("title", ""),
        subtask_reason=subtask.get("reason", ""),
        chain_memory=chain_memory,
        docs_text=docs_text,
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
        reasoning_trace = str(obj.get("reasoning_trace", "")).strip()
        return {
            "sub_query": str(obj.get("sub_query", subtask.get("title", ""))).strip(),
            "evidence": [str(e).strip() for e in obj.get("evidence", []) if str(e).strip()],
            "candidates": candidates,
            "confidence": min(1.0, max(0.0, float(obj.get("confidence", 0.0)))),
            "sources": [str(s).strip() for s in obj.get("sources", []) if str(s).strip()],
            "reasoning_trace": reasoning_trace,
        }
    except Exception as e:
        print(f"  [{subtask['id']}] extract_structured failed: {e}")
        return {
            "sub_query": subtask.get("title", ""),
            "evidence": [],
            "candidates": [],
            "confidence": 0.0,
            "sources": [],
            "reasoning_trace": "",
        }


async def _generate_verification_queries(flash_llm, question: str, subtask: Dict, candidates: List[str]) -> List[str]:
    cands_str = ", ".join(candidates)
    prompt = VERIFICATION_QUERY_GEN_PROMPT.format(
        question=question,
        subtask_title=subtask.get("title", ""),
        candidates=cands_str
    )
    try:
        resp = await flash_llm.ainvoke(prompt)
        obj = _safe_json_obj(str(resp.content).strip())
        raw_qs = obj.get("verify_queries", [])
        # flatten: LLM 有时会返回嵌套数组或把列表序列化成字符串
        flat_qs: List[str] = []
        for q in raw_qs:
            if isinstance(q, list):
                flat_qs.extend([str(x).strip() for x in q if str(x).strip()])
            else:
                s = str(q).strip()
                # 检测是否是 "['a', 'b']" 这种字符串化的列表
                if s.startswith("[") and s.endswith("]"):
                    try:
                        inner = json.loads(s.replace("'", '"'))
                        if isinstance(inner, list):
                            flat_qs.extend([str(x).strip() for x in inner if str(x).strip()])
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                if s:
                    flat_qs.append(s)
        return _dedup_strings(flat_qs)
    except Exception as e:
        print(f"  [{subtask['id']}] verification_query_gen error: {e}")
        fallback_queries: List[str] = []
        for candidate in candidates:
            fallback_queries.extend([
                f"{candidate} 百度百科",
                f"{candidate} wikipedia",
            ])
        return _dedup_strings(fallback_queries)

async def _verify_and_filter_candidates(
    flash_llm,
    question: str,
    subtask: Dict,
    candidates: List[str],
    verify_snippets: List[str],
) -> Tuple[List[str], List[str], str, str]:
    evidence_text = "\n\n".join(f"[V{i}] {s}" for i, s in enumerate(verify_snippets, 1))
    if not evidence_text:
        evidence_text = "（二次检索未获取到有效文档内容）"
    prompt = VERIFY_AND_FILTER_PROMPT.format(
        subtask_title=subtask.get("title", ""),
        subtask_reason=subtask.get("reason", ""),
        candidates=", ".join(candidates),
        verify_evidence=evidence_text,
    )
    try:
        resp = await flash_llm.ainvoke(prompt)
        obj = _safe_json_obj(str(resp.content).strip())
        analysis = str(obj.get("analysis", "")).strip()
        passed = [str(x).strip() for x in obj.get("passed_candidates", [])]
        rejected = [str(x).strip() for x in obj.get("rejected_candidates", [])]
        reason = str(obj.get("discard_reason", ""))
        
        # 兜底：只允许留下曾在原来 candidates 里的，以免发散产生臆想实体
        final_passed = [c for c in passed if any(c in orig or orig in c for orig in candidates)]
        if passed and not final_passed: 
            final_passed = passed # 若严格子串挂了，宽松保留
            
        return final_passed, rejected, reason, analysis
    except Exception as e:
        print(f"  [{subtask['id']}] verify_and_filter error: {e}")
        return candidates, [], f"过滤报错跳过:{e}", ""


def _build_candidate_branch_subtask(
    subtask: Dict,
    candidate: str,
    branch_index: int,
    candidate_packet: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """为候选分支构造独立子任务上下文，避免压缩与裁决阶段串线。"""
    branch_subtask = dict(subtask)
    branch_subtask["id"] = f"{subtask.get('id', 'ST')}.C{branch_index}"
    title = subtask.get("title", "").strip()
    reason = subtask.get("reason", "").strip()
    branch_subtask["title"] = f"{title}｜候选核验：{candidate}" if title else f"候选核验：{candidate}"
    branch_hint = f"当前分支只验证候选「{candidate}」是否满足该子任务要求。"
    packet_context = str((candidate_packet or {}).get("curated_context", "")).strip()
    if packet_context:
        branch_hint = f"{branch_hint}\n{packet_context}"
    branch_subtask["reason"] = f"{reason}\n{branch_hint}".strip() if reason else branch_hint
    return branch_subtask


async def _verify_candidate_branch(
    flash_llm,
    question: str,
    subtask: Dict,
    candidate: str,
    branch_index: int,
    task_packet: Dict[str, Any] | None,
    searcher,
    fetcher,
    max_docs: int = 5,
    per_query_results: int = 2,
) -> Dict[str, Any]:
    """
    单候选独立验证分支：
    query 生成、search/fetch/compress、裁决全部在本分支内闭环完成。
    """
    candidate_packet = build_candidate_packet(task_packet or {}, candidate)
    branch_subtask = _build_candidate_branch_subtask(subtask, candidate, branch_index, candidate_packet)

    generated_queries = await _generate_verification_queries(
        flash_llm, question, branch_subtask, [candidate]
    )
    branch_queries = _dedup_strings(generated_queries)[: int(candidate_packet.get("budget", {}).get("max_queries", 4))]
    print(f"    [{branch_subtask['id']}] branch_queries({candidate})={branch_queries}")

    # 候选分支不与其他候选共享 existing_urls，避免 URL 被先到先得地“抢走”。
    branch_docs, branch_used_qs, branch_snippets = await _search_fetch_compress(
        branch_queries,
        searcher,
        fetcher,
        flash_llm,
        question,
        branch_subtask,
        set(),
        max_docs=int(candidate_packet.get("budget", {}).get("max_docs", max_docs)),
        per_query_results=per_query_results,
    )
    passed, rejected, discard_reason, analysis = await _verify_and_filter_candidates(
        flash_llm, question, branch_subtask, [candidate], branch_snippets
    )
    passed = _dedup_strings(passed)
    rejected = _dedup_strings(rejected)
    branch_sources = _dedup_strings([getattr(doc, "url", "") for doc in branch_docs if getattr(doc, "url", "")])

    print(
        f"    [{branch_subtask['id']}] branch_verdict candidate={candidate} "
        f"passed={passed} rejected={rejected} docs={len(branch_docs)}"
    )
    if analysis:
        print(f"    [{branch_subtask['id']}] branch_analysis: {analysis}")
    if discard_reason:
        print(f"    [{branch_subtask['id']}] branch_discard_reason: {discard_reason}")

    return {
        "candidate": candidate,
        "queries": branch_queries,
        "used_queries": branch_used_qs,
        "docs": branch_docs,
        "snippets": branch_snippets,
        "passed_candidates": passed,
        "rejected_candidates": rejected,
        "discard_reason": discard_reason,
        "analysis": analysis,
        "sources": branch_sources,
        "task_packet": candidate_packet,
    }


async def _verify_candidates_in_isolation(
    flash_llm,
    question: str,
    subtask: Dict,
    task_packet: Dict[str, Any] | None,
    candidates: List[str],
    searcher,
    fetcher,
) -> Dict[str, Any]:
    """对每个 candidate 启动独立验证分支，最后再融合幸存者与分支推理。"""
    unique_candidates = _dedup_strings(candidates)
    if not unique_candidates:
        return {
            "passed_candidates": [],
            "rejected_candidates": [],
            "discard_reason": "无候选可供分支核验。",
            "branch_details": [],
            "docs": [],
            "used_queries": [],
            "sources": [],
            "reasoning_lines": [],
            "evidence_lines": [],
        }

    branch_tasks = [
        _verify_candidate_branch(
            flash_llm=flash_llm,
            question=question,
            subtask=subtask,
            candidate=candidate,
            branch_index=idx,
            task_packet=task_packet,
            searcher=searcher,
            fetcher=fetcher,
        )
        for idx, candidate in enumerate(unique_candidates, 1)
    ]
    branch_results = await asyncio.gather(*branch_tasks)

    passed_candidates: List[str] = []
    rejected_candidates: List[str] = []
    discard_reasons: List[str] = []
    reasoning_lines: List[str] = []
    evidence_lines: List[str] = []
    merged_docs: List[Document] = []
    merged_doc_urls: Set[str] = set()
    used_queries: List[str] = []
    sources: List[str] = []
    branch_details: List[Dict[str, Any]] = []

    for branch in branch_results:
        _extend_unique_docs(merged_docs, branch.get("docs", []), merged_doc_urls)
        used_queries.extend(branch.get("used_queries", []))

        branch_passed = _dedup_strings(branch.get("passed_candidates", []))
        branch_rejected = _dedup_strings(branch.get("rejected_candidates", []))
        candidate = branch.get("candidate", "")
        analysis = str(branch.get("analysis", "")).strip()
        discard_reason = str(branch.get("discard_reason", "")).strip()
        branch_sources = _dedup_strings(branch.get("sources", []))
        sources.extend(branch_sources)

        if branch_passed:
            passed_candidates.extend(branch_passed)
            outcome = f"保留 {', '.join(branch_passed)}"
            why = analysis or "分支证据与子任务约束一致。"
            reasoning_lines.append(f"[{candidate}] {outcome}。{why}")
            evidence_lines.append(f"分支核验[{candidate}]：{why}")
        else:
            if branch_rejected:
                rejected_candidates.extend(branch_rejected)
            else:
                rejected_candidates.append(candidate)
            why = analysis or discard_reason or "独立分支中未找到足以支持该候选的交叉证据。"
            reasoning_lines.append(f"[{candidate}] 剔除。{why}")

        if discard_reason:
            discard_reasons.append(f"{candidate}: {discard_reason}")

        branch_details.append({
            "candidate": candidate,
            "queries": branch.get("queries", []),
            "passed_candidates": branch_passed,
            "rejected_candidates": branch_rejected,
            "discard_reason": discard_reason,
            "analysis": analysis,
            "sources": branch_sources,
        })

    return {
        "passed_candidates": _dedup_strings(passed_candidates),
        "rejected_candidates": _dedup_strings(rejected_candidates),
        "discard_reason": "；".join(_dedup_strings(discard_reasons)),
        "branch_details": branch_details,
        "docs": merged_docs,
        "used_queries": _dedup_strings(used_queries),
        "sources": _dedup_strings(sources),
        "reasoning_lines": reasoning_lines,
        "evidence_lines": _dedup_strings(evidence_lines),
    }

async def _failure_reflect_subtask(
    flash_llm, question: str, subtask: Dict,
    used_queries: List[str], evidence: List[str],
    candidates: List[str], discard_reason: str
) -> Tuple[str, str, List[str]]:
    ev_str = "\n".join(f"- {e}" for e in evidence[:10]) or "（无实质证据抽回）"
    prompt = FAILURE_REFLECT_PROMPT.format(
        subtask_title=subtask.get("title", ""),
        used_queries=", ".join(used_queries) if used_queries else "（无）",
        evidence=ev_str,
        candidates=", ".join(candidates) if candidates else "（在抽取环节就空手而归）",
        discard_reason=discard_reason or "一无所获导致断索",
    )
    try:
        resp = await flash_llm.ainvoke(prompt)
        obj = _safe_json_obj(str(resp.content).strip())
        reflection = str(obj.get("reflection", ""))
        tactic_advice = str(obj.get("tactic_advice", ""))
        suggested_queries = [str(q).strip() for q in obj.get("suggested_queries", []) if str(q).strip()]
        return reflection, tactic_advice, suggested_queries
    except Exception as e:
        print(f"  [{subtask['id']}] failure_reflect error: {e}")
        return "", "", []


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
    brief: Dict | None = None,
    plan_review: Dict | None = None,
    memory: ExecutionMemory | None = None,
    group_index: int = 0,
) -> Tuple[str, List[Document], List[str], Dict, Dict[str, Any], Dict[str, Any]]:
    """
    处理单个子任务完整流程（含 Rewriter + 自查重试）。
    返回 (subtask_id, new_docs, used_queries, structured_result, task_packet, worker_artifact)。
    """
    import time as _time
    _step_start = _time.time()
    
    st_id = subtask["id"]
    
    # ── 预处理 title/reason：用前序 candidates 替换 [STx] 占位符 ──
    resolved_subtask = dict(subtask)
    st_title = resolved_subtask.get("title", "")
    st_reason = resolved_subtask.get("reason", "")
    deps = resolved_subtask.get("depends_on", [])
    _invalid = {"未找到相关文档。", "提取失败。", "未执行。", "未生成有效查询。"}
    if deps and prev_results:
        for dep_id in deps:
            r = prev_results.get(dep_id)
            if not isinstance(r, dict):
                continue
            best = (r.get("candidates") or [""])[0]
            if best and best not in _invalid:
                pattern = rf'\[?{dep_id}\]?|[（\(]?{dep_id}[）\)]?'
                st_title = re.sub(pattern, best, st_title, flags=re.IGNORECASE)
                st_reason = re.sub(pattern, best, st_reason, flags=re.IGNORECASE)
    resolved_subtask["title"] = st_title
    resolved_subtask["reason"] = st_reason
    
    print(f"\n  ── 执行子任务 [{st_id}] {resolved_subtask.get('title', '')} ──")

    # ── 注入规划阶段的猜测答案（仅当前子任务的猜测，不泄漏最终答案） ──
    guess_answer = resolved_subtask.get("guess_answer", "")

    all_new_docs: List[Document] = []
    all_new_doc_urls: Set[str] = set()
    all_used_queries: List[str] = []
    all_snippets: List[str] = []
    retry_context: str = ""  # 上轮 self_check 的诊断经验
    latest_task_packet: Dict[str, Any] = {}
    result: Dict = {
        "sub_query": resolved_subtask.get("title", ""),
        "evidence": [],
        "candidates": [],
        "confidence": 0.0,
        "sources": [],
        "reasoning_trace": "",
    }

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"  [{st_id}] === 自查重试 第{attempt}次 ===")

        latest_task_packet = build_task_packet(
            question=question,
            brief=brief or {},
            subtask=resolved_subtask,
            prev_results=prev_results,
            memory=memory,
            plan_review=plan_review or {},
            retry_context=retry_context,
        )
        memory_context = latest_task_packet.get("curated_context", "")
        if guess_answer:
            plan_guess_block = f"【规划阶段对本子任务的初步猜测（仅供参考，需搜索验证）】\n  {guess_answer}"
            memory_context = (plan_guess_block + "\n\n" + memory_context) if memory_context else plan_guess_block
        if attempt == 0:
            print(f"  [{st_id}] task_packet constraints={latest_task_packet.get('constraints', [])}")
            if latest_task_packet.get("instruction"):
                print(f"  [{st_id}] task_packet instruction={latest_task_packet.get('instruction', '')}")

        if attempt == 0:
            # ── Step 1: 首次尝试，生成极简零散实体的纯 Anchor 查询 ──
            raw_queries = await _generate_queries(
                flash_llm, question, resolved_subtask, prev_results,
                brief=brief, task_packet=latest_task_packet, retry_context="",
            )
            raw_queries = _dedup_queries(raw_queries, {q.lower() for q in query_history})
            print(f"  [{st_id}] generated_queries={len(raw_queries)}")
            for i, q in enumerate(raw_queries, 1):
                print(f"  [{st_id}]   gen_{i}: {q}")
            
            final_queries = raw_queries
            print(f"  [{st_id}] 首次尝试跳过 optimize，直接使用 {len(final_queries)} 条基准 query")
        else:
            # ── Step 2: 重试时，强逼大模型阅读诊断记录并执行 reflect + expand ──
            print(f"  [{st_id}] 重试触发：绕过 generate_queries，直接调用 optimize(带诊断诊断上下文) 进行泛化")
            
            st_title_temp = resolved_subtask.get("title", "")
            st_reason_temp = resolved_subtask.get("reason", "")
            enriched_reason = f"{st_reason_temp}\n\n【⚠️本次为重试，必须吸取上轮失败教训】：\n{retry_context}"
            temp_subtask = {**resolved_subtask, "reason": enriched_reason, "queries": raw_queries}
            
            final_queries = await _optimize_queries_for_subtask(
                flash_llm, question, temp_subtask, key_entities, query_history
            )
            print(f"  [{st_id}] optimized_queries={len(final_queries)}")
            for i, q in enumerate(final_queries, 1):
                print(f"  [{st_id}]   opt_{i}: {q}")

        # ── Step 3: Search → Fetch → Compress ──
        retrieval_budget = int(latest_task_packet.get("budget", {}).get("max_docs", 6))
        if final_queries:
            new_docs, used_qs, new_snippets = await _search_fetch_compress(
                final_queries, searcher, fetcher, flash_llm, question, resolved_subtask,
                existing_urls, max_docs=retrieval_budget,
            )
        else:
            new_docs, used_qs, new_snippets = [], [], []

        _extend_unique_docs(all_new_docs, new_docs, all_new_doc_urls)
        for d in new_docs:
            if getattr(d, "url", None):
                existing_urls.add(d.url)
        all_used_queries.extend(used_qs)
        all_snippets.extend(new_snippets)
        print(f"  [{st_id}] docs={len(new_docs)}, snippets={len(new_snippets)}")

        # ── Step 4: 提取结构化发现 ──
        result = await _extract_structured_findings(
            flash_llm, question, resolved_subtask, all_snippets, memory_context
        )
        candidates = result.get('candidates', [])
        print(f"  [{st_id}] initial candidates: {candidates}")
        print(f"  [{st_id}] confidence: {result.get('confidence', 0)}")
        
        # ── Step 4.5: 融合规划猜测的 candidates ──
        if guess_answer and guess_answer not in candidates:
            candidates.append(guess_answer)
            result['candidates'] = candidates
            print(f"  [{st_id}] 追加规划猜测候选: {guess_answer}")

        # ── Step 5: 后置验证 Queries 路径 (双规核查) ──
        discard_reason = ""
        if candidates:
            # 5.1 候选分支独立核验：每个 candidate 单独 search/fetch/compress，再统一融合
            branch_verification = await _verify_candidates_in_isolation(
                flash_llm=flash_llm,
                question=question,
                subtask=resolved_subtask,
                task_packet=latest_task_packet,
                candidates=candidates,
                searcher=searcher,
                fetcher=fetcher,
            )
            passed_cands = branch_verification.get("passed_candidates", [])
            rejected_cands = branch_verification.get("rejected_candidates", [])
            discard_reason = branch_verification.get("discard_reason", "")
            branch_details = branch_verification.get("branch_details", [])
            branch_reasoning_lines = branch_verification.get("reasoning_lines", [])
            branch_evidence_lines = branch_verification.get("evidence_lines", [])
            branch_sources = branch_verification.get("sources", [])
            branch_docs = branch_verification.get("docs", [])
            branch_used_qs = branch_verification.get("used_queries", [])

            _extend_unique_docs(all_new_docs, branch_docs, all_new_doc_urls)
            for d in branch_docs:
                if getattr(d, "url", None):
                    existing_urls.add(d.url)
            all_used_queries.extend(branch_used_qs)

            if branch_details:
                print(f"  [{st_id}] candidate_branches={len(branch_details)}")
                for detail in branch_details:
                    print(
                        f"  [{st_id}]   branch {detail.get('candidate')}: "
                        f"passed={detail.get('passed_candidates', [])} "
                        f"rejected={detail.get('rejected_candidates', [])}"
                    )

            print(f"  [{st_id}] 经二次交叉网搜校验保留下来的有效候选: {passed_cands}")
            if rejected_cands:
                print(f"  [{st_id}] ⚠️ 被验证结果剔除的候选: {rejected_cands}")
                print(f"  [{st_id}] 剔除原因: {discard_reason}")
            
            # 回写清洗过的名单
            result["candidates"] = passed_cands
            result["branch_details"] = branch_details
            if branch_sources:
                result["sources"] = _dedup_strings(result.get("sources", []) + branch_sources)
            if branch_evidence_lines:
                result["evidence"] = _dedup_strings(result.get("evidence", []) + branch_evidence_lines)
            if branch_reasoning_lines:
                branch_reasoning = "【候选分支独立核验】\n" + "\n".join(branch_reasoning_lines)
                existing_trace = result.get("reasoning_trace", "").strip()
                result["reasoning_trace"] = (
                    f"{existing_trace}\n\n{branch_reasoning}".strip()
                    if existing_trace else branch_reasoning
                )
        else:
            discard_reason = "初代提取环节彻底崩盘，连个候选实体的影子都没抓到。"

        # ── Step 6: 结果裁定与反思路径 (Reflect 准备下一次重试) ──
        if result.get("candidates"):
            print(f"  [{st_id}] 验证幸存结束，已成功锁定事实答案。")
            break # 退出重试循环！此子任务盖棺定论！
        else:
            if attempt < max_retries:
                print(f"  [{st_id}] 本轮宣告失败，启动 Reflect 反思重整评估...")
                reflection, tactic, suggested = await _failure_reflect_subtask(
                    flash_llm, question, resolved_subtask, raw_queries, result.get("evidence", []),
                    candidates, discard_reason
                )
                
                # 为下一次 Optimize 重试阶段装配教训和锦囊
                retry_parts = []
                if reflection: retry_parts.append(f"【战术反思诊断】: {reflection}")
                if tactic: retry_parts.append(f"【执行层指导令】: {tactic}")
                if suggested: retry_parts.append(f"【建议突破查询】: {', '.join(suggested)}")
                retry_context = "\n".join(retry_parts)
                print(f"  [{st_id}] retry_context generated ({len(retry_context)} chars)")
        # 最后一次尝试若仍全军覆没 → 默认接受当前空弹匣结果

    # ── 记录到执行记忆 ──
    if memory:
        candidates_final = result.get('candidates', [])
        record = StepRecord(
            step_id=st_id,
            title=resolved_subtask.get("title", ""),
            group_index=group_index,
            candidates=candidates_final,
            best_answer=candidates_final[0] if candidates_final else "",
            evidence=result.get('evidence', []),
            reasoning_trace=result.get('reasoning_trace', ''),
            confidence=result.get('confidence', 0.0),
            queries_used=all_used_queries,
            attempt_count=attempt + 1,
            success=bool(candidates_final),
            failure_reason=discard_reason if not candidates_final else "",
            start_time=_step_start,
            end_time=_time.time(),
        )
        memory.record_step(record)
        print(f"  [{st_id}] 已记录到执行记忆 (success={record.success}, dur={record.duration_s:.1f}s)")

    worker_artifact = {
        "status": "success" if result.get("candidates") else "failed",
        "task_packet": latest_task_packet,
        "queries_used": _dedup_strings(all_used_queries),
        "doc_count": len(all_new_docs),
        "snippet_count": len(all_snippets),
        "branch_details": result.get("branch_details", []),
        "failure_reason": discard_reason if not result.get("candidates") else "",
        "best_candidate": (result.get("candidates") or [""])[0] if result.get("candidates") else "",
        "confidence": result.get("confidence", 0.0),
    }

    return st_id, all_new_docs, all_used_queries, result, latest_task_packet, worker_artifact


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
        plan_review: Dict[str, Any] = dict(state.get("plan_review", {}) or {})
        task_packets: Dict[str, Any] = dict(state.get("task_packets", {}) or {})
        worker_artifacts: Dict[str, Any] = dict(state.get("worker_artifacts", {}) or {})

        # ── 恢复或创建执行记忆 ──
        raw_mem = state.get("execution_memory") or {}
        if raw_mem:
            memory = ExecutionMemory.from_dict(raw_mem)
            memory.iteration += 1
            print(f"[execute_subtasks] 恢复执行记忆: {len(memory.steps)} 步骤, iter={memory.iteration}")
        else:
            memory = ExecutionMemory()
            print("[execute_subtasks] 初始化新的执行记忆")

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

        print(f"[execute_subtasks] subtasks={len(subtasks)}, groups={len(parallel_groups)}, problem_type={brief.get('problem_type', 'unknown')}")
        for gi, group in enumerate(parallel_groups):
            print(f"  group_{gi}: {group}")

        all_new_docs: List[Document] = []
        all_used_queries: List[str] = []

        # ── 按层执行 ──
        for gi, group in enumerate(parallel_groups):
            print(f"\n══ 执行并行组 {gi} / {len(parallel_groups)-1}: {group} ══")

            # 构建子任务执行闭包
            async def _run_subtask(st_id: str, _gi: int = gi):
                st = subtask_map.get(st_id)
                if not st:
                    empty_result = {
                        "sub_query": "", "evidence": [],
                        "candidates": [], 
                        "confidence": 0.0, "sources": [],
                    }
                    return st_id, [], [], empty_result, {}, {}

                # 如果已有结果，跳过
                if subtask_findings.get(st_id):
                    print(f"  [{st_id}] 已有结果，跳过。")
                    return st_id, [], [], subtask_findings[st_id], task_packets.get(st_id, {}), worker_artifacts.get(st_id, {})

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
                    brief=brief,
                    plan_review=plan_review,
                    memory=memory,
                    group_index=_gi,
                )

            # 组内并行
            results = await asyncio.gather(*[_run_subtask(st_id) for st_id in group])

            for st_id, new_docs, used_qs, result, task_packet, worker_artifact in results:
                subtask_findings[st_id] = result
                if task_packet:
                    task_packets[st_id] = task_packet
                if worker_artifact:
                    worker_artifacts[st_id] = worker_artifact
                all_new_docs.extend(new_docs)
                all_used_queries.extend(used_qs)
                # 更新 existing_urls 供后续组使用
                for d in new_docs:
                    if getattr(d, "url", None):
                        existing_urls.add(d.url)

            # ── 组间推理链摘要（LLM 压缩） ──
            group_records = [memory.steps[sid] for sid in group if sid in memory.steps]
            if group_records:
                group_summary = await generate_group_summary(
                    flash_llm=_flash,
                    question=question,
                    group_records=group_records,
                )
                memory.set_group_summary(gi, group_summary)
                print(f"\n  ── 组 {gi} LLM 摘要 ──")
                print(f"  {group_summary[:300]}")
            else:
                print(f"\n  ── 组 {gi} 无新记录（均已跳过或全部失败） ──")

            # 打印各子任务推理链
            print(f"\n  ── 组 {gi} 推理链明细 ──")
            for st_id in group:
                r = subtask_findings.get(st_id, {})
                if isinstance(r, dict):
                    cands = r.get('candidates', [])
                    reasoning = r.get('reasoning_trace', '')
                    best = cands[0] if cands else '(未确定)'
                    print(f"  [{st_id}] → {best}")
                    if reasoning:
                        print(f"    推理: {reasoning[:200]}")
                    ev = r.get('evidence', [])
                    if ev:
                        print(f"    证据: {'; '.join(ev[:2])}")

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
                reasoning = result.get('reasoning_trace', '')
                print(f"  [{st_id}] best={best[:80]} cands={cands} conf={result.get('confidence', 0)}")
                if reasoning:
                    print(f"    reasoning: {reasoning[:150]}")
            else:
                print(f"  [{st_id}] {str(result)[:100]}...")

        # ── 打印执行记忆状态 ──
        memory.print_summary()

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
            "execution_memory": memory.to_dict(),
            "task_packets": task_packets,
            "worker_artifacts": worker_artifacts,
            "messages": [msg],
        }

    return execute_subtasks
