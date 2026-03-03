"""deepresearch/nodes/query_optimize.py

两阶段流水线：
1) Query Reflection  — 逐条识别查询的 4 类问题并重写
   - 信息歧义（Information Ambiguity）
   - 语义歧义（Semantic Ambiguity）
   - 复杂需求（Complex Requirements）
   - 过度具体（Overly Specific）
2) Query Rollout/Expansion — 基于重写结果扩展出多样化变体
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import AIMessage

try:
    from ..prompt_loader import load_prompt
except ImportError:
    from deepresearch.prompt_loader import load_prompt

try:
    from ..state import DeepResearchState
except ImportError:
    # 直接运行时的兼容
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deepresearch.state import DeepResearchState


# ───────── 工具函数 ─────────

def _safe_json_obj(text: str) -> Dict:
    """从 LLM 输出中提取 JSON 对象。"""
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def _parse_rollout_queries(text: str, n_rollout: int) -> List[str]:
    """
    解析 query_rollout 输出：优先提取 <begin>...<end> 标记内的内容，
    否则尝试 JSON 数组，最后按行分割。
    """
    def _clean_query(q: str) -> str:
        """清理单条 query：移除编号前缀、多余空白等。"""
        q = q.strip()
        # 移除 "query_1:", "1.", "1)" 等前缀
        q = re.sub(r"^(?:query[_\s]*\d+[:\.\)]\s*|\d+[:\.\)]\s*)", "", q, flags=re.IGNORECASE)
        return q.strip()

    # 1) 优先 <begin>...<end> 格式
    m = re.search(r"<begin>([\s\S]*?)(?:<end>|$)", text, re.IGNORECASE)
    if m:
        block = m.group(1).strip()
        lines = [_clean_query(ln) for ln in block.split("\n") if ln.strip()]
        lines = [ln for ln in lines if ln]  # 过滤空行
        return lines[:n_rollout] if lines else []

    # 2) 尝试 JSON 数组
    arr_match = re.search(r"\[[\s\S]*?\]", text)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            return [_clean_query(str(x)) for x in arr if str(x).strip()][:n_rollout]
        except json.JSONDecodeError:
            pass

    # 3) 按行分割
    lines = [_clean_query(ln) for ln in text.strip().split("\n") if ln.strip()]
    lines = [ln for ln in lines if ln]
    return lines[:n_rollout]


# ───────── Prompt 模板（参考 OAgents search_prompts.yaml） ─────────

BATCH_REFLECTION_PROMPT = load_prompt("query_optimize.yaml", "batch_reflection_prompt")
QUERY_ROLLOUT_PROMPT = load_prompt("query_optimize.yaml", "query_rollout_prompt")
RESULT_REFLECTION_PROMPT = load_prompt("query_optimize.yaml", "result_reflection_prompt")


# ───────── 核心函数 ─────────

def _parse_batch_reflection(text: str, raw_queries: List[str]) -> List[Tuple[str, str]]:
    """
    解析批量 reflect 的 JSON 数组输出，返回 [(analysis, augmented), ...]。
    支持 augmented 字段用 | 分隔多条拆分查询。
    """
    # 尝试提取 JSON 数组
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            results = []
            for i, item in enumerate(arr):
                analysis = str(item.get("analysis", "")).strip()
                augmented = str(item.get("augmented", "")).strip()
                orig = raw_queries[i] if i < len(raw_queries) else ""
                if not augmented:
                    augmented = orig
                # 支持 | 分隔的多查询拆分
                if "|" in augmented:
                    parts = [p.strip() for p in augmented.split("|") if p.strip()]
                    for p in parts:
                        results.append((analysis, p))
                else:
                    results.append((analysis, augmented))
            return results
        except (json.JSONDecodeError, IndexError):
            pass
    # fallback：原样返回
    return [("", q) for q in raw_queries]


async def _reflect_batch(llm, question: str, queries: List[str]) -> List[Tuple[str, str]]:
    """
    批量反思所有 queries（一次 LLM 调用搞定），返回 [(analysis, augmented_query), ...]。
    注意：结果数量可能 >= 输入数量（因为 Complex Requirements 会拆分为多条）。
    """
    if not queries:
        return []

    queries_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    prompt = BATCH_REFLECTION_PROMPT.format(question=question, queries_block=queries_block)

    try:
        resp = await llm.ainvoke(prompt)
        results = _parse_batch_reflection(str(resp.content), queries)
        if not results:
            return [("", q) for q in queries]
        return results
    except Exception as e:
        print(f"[query_reflect_batch] failed: {e}")
        return [("", q) for q in queries]


async def reflect_search_results(
    llm, query: str, results: List[Dict]
) -> List[Dict]:
    """
    对搜索结果进行相关性评分（参考 OAgents SearchReflector.result_reflect）。
    
    Args:
        llm: LLM 实例
        query: 搜索查询
        results: 搜索结果列表，每个结果需包含 idx, title, snippet
        
    Returns:
        带有评分的结果列表，按 overall_score 降序排列
    """
    if not results:
        return []
    
    # 构建结果文本
    results_lines = []
    for r in results:
        idx = r.get("idx", 0)
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        results_lines.append(f"idx: {idx}\ntitle: {title}\nsnippet: {snippet}")
    results_text = "\n\n".join(results_lines)
    
    prompt = RESULT_REFLECTION_PROMPT.format(query=query, results_text=results_text)
    
    try:
        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        
        # 解析 JSON 数组
        arr_match = re.search(r"\[[\s\S]*?\]", raw)
        if arr_match:
            scores = json.loads(arr_match.group(0))
            
            # 将评分合并回原结果
            score_map = {s.get("idx"): s for s in scores}
            for r in results:
                idx = r.get("idx", 0)
                if idx in score_map:
                    r["similarity_score"] = score_map[idx].get("similarity_score", 5)
                    r["overall_score"] = score_map[idx].get("overall_score", 5)
                else:
                    r["similarity_score"] = 5
                    r["overall_score"] = 5
            
            # 按 overall_score 降序排序
            results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
            return results
    except Exception as e:
        print(f"[result_reflect] failed: {e}")
    
    # 失败时返回原结果
    return results


async def _rollout_query(
    llm, question: str, query: str, key_entities: List[str], history: List[str], n_rollout: int
) -> List[str]:
    """
    基于单条 query 扩展出多条变体（参考 OAgents SearchReflector.query_rollout）。
    """
    history_text = "\n".join(f"- {q}" for q in history[-20:]) if history else "(none)"
    entities_text = ", ".join(key_entities) if key_entities else "(not specified)"

    prompt = QUERY_ROLLOUT_PROMPT.format(
        question=question,
        query=query,
        key_entities=entities_text,
        history_text=history_text,
        roll_out=n_rollout,
    )

    try:
        resp = await llm.ainvoke(prompt)
        queries = _parse_rollout_queries(str(resp.content), n_rollout)
        # 始终把原始 query 加到末尾作为兜底
        if query not in queries:
            queries.append(query)
        return queries
    except Exception as e:
        print(f"[query_rollout] failed: {e}")
        return [query]


async def _run_reflect_rollout(state: DeepResearchState, llm, flash_llm=None) -> Dict[str, Any]:
    """公共逻辑：批量 reflect + 并行 rollout。flash_llm 用于轻量推理，默认退回 llm。"""
    import asyncio

    _llm = flash_llm or llm          # reflect/rollout 都用 flash
    raw_queries: List[str] = state.get("queries", []) or []
    question: str = state.get("question", "")
    brief = state.get("research_brief", {}) or {}
    query_history: List[str] = state.get("query_history", []) or []
    key_entities: List[str] = brief.get("key_entities", [])

    if not raw_queries:
        print("[query_optimize] 无待优化查询，跳过。")
        return {
            "queries": [],
            "messages": [AIMessage(content="[query_optimize] 无查询，跳过。")],
            "_meta": {
                "raw_queries": 0,
                "refined_queries": 0,
                "expanded_queries": 0,
            },
        }

    print(f"[query_optimize] input_queries={len(raw_queries)}, history={len(query_history)}")
    for i, q in enumerate(raw_queries, 1):
        print(f"[query_optimize] raw_{i}: {q}")

    # Step 1: Batch Reflection（一次 LLM 调用）
    reflect_results = await _reflect_batch(_llm, question, raw_queries)
    refined_queries: List[str] = []
    for (analysis, augmented), orig in zip(reflect_results, raw_queries):
        if analysis:
            print(f"[query_optimize] reflect: '{orig[:40]}' -> '{augmented[:40]}'")
        else:
            print(f"[query_optimize] reflect: '{orig[:40]}' -> (unchanged)")
        refined_queries.append(augmented)

    # Step 2: Parallel Rollout（asyncio.gather 并发）
    rollout_per_query = max(2, min(4, 10 // len(refined_queries))) if refined_queries else 2

    async def _do_rollout(rq: str) -> List[str]:
        variants = await _rollout_query(_llm, question, rq, key_entities, query_history, rollout_per_query)
        print(f"[query_optimize] rollout '{rq[:30]}' -> {len(variants)} variants")
        return variants

    # 创建并发任务列表：每个refined query独立发起一次LLM rollout调用
    rollout_tasks = [_do_rollout(rq) for rq in refined_queries]
    # 使用asyncio.gather并发执行所有rollout任务，提高效率
    rollout_results = await asyncio.gather(*rollout_tasks)

    expanded_queries: List[str] = []
    for variants in rollout_results:
        expanded_queries.extend(variants)

    # Step 3: Merge + dedup
    all_candidates = refined_queries + expanded_queries
    history_set = {q.strip().lower() for q in query_history}
    final_queries: List[str] = []
    seen: set = set()

    for q in all_candidates:
        q_clean = q.strip()
        q_norm = q_clean.lower()
        if not q_clean:
            continue
        if q_norm in seen:
            continue
        if q_norm in history_set:
            continue
        seen.add(q_norm)
        final_queries.append(q_clean)

    if len(final_queries) < 3:
        for q in raw_queries:
            q_clean = q.strip()
            q_norm = q_clean.lower()
            if q_norm not in seen and q_clean:
                seen.add(q_norm)
                final_queries.append(q_clean)

    final_queries = final_queries[:12]

    print(f"[query_optimize] final_queries={len(final_queries)}")
    for i, q in enumerate(final_queries, 1):
        print(f"[query_optimize] final_{i}: {q}")

    return {
        "queries": final_queries,
        "_meta": {
            "raw_queries": len(raw_queries),
            "refined_queries": len(refined_queries),
            "expanded_queries": len(expanded_queries),
        },
    }


def make_query_optimize_node_reflect_rollout(llm, flash_llm=None) -> Callable[[DeepResearchState], DeepResearchState]:
    """版本1：仅 query reflect + rollout。flash_llm 用于轻量推理（默认 qwen3.5-flash）。"""

    async def query_optimize(state: DeepResearchState) -> DeepResearchState:
        print("\n============ query_optimize(v1: reflect+rollout) 阶段 ============")
        out = await _run_reflect_rollout(state, llm, flash_llm=flash_llm)
        meta = out.pop("_meta", {})
        msg = AIMessage(
            content=(
                f"[query_optimize:v1] 反思优化 {meta.get('raw_queries', 0)}→{meta.get('refined_queries', 0)} 条，"
                f"扩展至 {meta.get('expanded_queries', 0)} 条，去重后最终 {len(out.get('queries', []))} 条。"
            )
        )
        out["messages"] = [msg]
        return out

    return query_optimize


def make_query_optimize_node_full(llm, flash_llm=None) -> Callable[[DeepResearchState], DeepResearchState]:
    """版本2：完整链路（reflect + rollout + result_reflection）。flash_llm 用于轻量推理。"""

    async def query_optimize_full(state: DeepResearchState) -> DeepResearchState:
        print("\n============ query_optimize(v2: full) 阶段 ============")
        out = await _run_reflect_rollout(state, llm, flash_llm=flash_llm)
        meta = out.pop("_meta", {})

        # 可选：对 state 中已有的搜索结果做 result_reflection 评分
        # 约定输入：state["search_results_by_query"] = {query: [{idx,title,snippet,...}, ...]}
        reranked_by_query: Dict[str, List[Dict]] = {}
        sr_by_query = state.get("search_results_by_query", {}) or {}
        if isinstance(sr_by_query, dict):
            for q, results in sr_by_query.items():
                if isinstance(results, list) and results:
                    reranked_by_query[q] = await reflect_search_results(llm, q, results)

        msg = AIMessage(
            content=(
                f"[query_optimize:v2] 反思优化 {meta.get('raw_queries', 0)}→{meta.get('refined_queries', 0)} 条，"
                f"扩展至 {meta.get('expanded_queries', 0)} 条，去重后最终 {len(out.get('queries', []))} 条，"
                f"result_reflection 处理 {len(reranked_by_query)} 个查询。"
            )
        )

        out["messages"] = [msg]
        if reranked_by_query:
            out["reranked_search_results_by_query"] = reranked_by_query
        return out

    return query_optimize_full


def make_query_optimize_node(llm, flash_llm=None) -> Callable[[DeepResearchState], DeepResearchState]:
    """兼容旧调用：默认返回 v1（reflect+rollout）。flash_llm 用于轻量推理。"""
    return make_query_optimize_node_reflect_rollout(llm, flash_llm=flash_llm)


# ───────── 测试入口 ─────────

if __name__ == "__main__":
    import asyncio
    import sys
    import os

    # 确保能导入 deepresearch 包
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from deepresearch.state import DeepResearchState

    load_dotenv()

    async def _demo():
        import time

        # flash 模型用于 reflect/rollout（快且便宜）
        flash_llm = ChatOpenAI(
            model="qwen3.5-flash",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
        )
        # 主模型（测试时也用 flash，实际可用 plus）
        llm = flash_llm

        # 模拟输入状态
        test_question = (
            "A Swiss artist who was a leader in the Dada movement is known for a painted "
            "plaster sculpture that appeared on the cover of a Tina Turner album. "
            "What year did this artist pass away?"
        )

        test_queries = [
            "Swiss Dada artist plaster sculpture Tina Turner album cover",
            "Tina Turner album painted sculpture cover artist",
            "Dada movement leader Swiss sculptor death year",
        ]

        state: DeepResearchState = {
            "question": test_question,
            "queries": test_queries,
            "query_history": [],
            "research_brief": {
                "key_entities": ["Dada movement", "Swiss artist", "Tina Turner", "plaster sculpture"],
            },
            "messages": [],
        }

        print("=" * 60)
        print("Query Optimize Node 测试")
        print("=" * 60)
        print(f"\n原始问题:\n{test_question}\n")
        print(f"输入查询 ({len(test_queries)} 条):")
        for i, q in enumerate(test_queries, 1):
            print(f"  {i}. {q}")
        print()

        # 版本1：仅 reflect + rollout（batch + parallel）
        t0 = time.time()
        node_v1 = make_query_optimize_node_reflect_rollout(llm, flash_llm=flash_llm)
        new_state_v1 = await node_v1(state)
        elapsed = time.time() - t0
        print(f"\n⏱  v1 耗时: {elapsed:.1f}s")

        print("\n" + "=" * 60)
        print("v1 最终输出查询（reflect + rollout）:")
        print("=" * 60)
        final_queries = new_state_v1.get("queries", [])
        for i, q in enumerate(final_queries, 1):
            print(f"  {i}. {q}")

        print(f"\n共 {len(final_queries)} 条查询")

        # ─────── 测试 Result Reflection ───────
        print("\n" + "=" * 60)
        print("Result Reflection 测试")
        print("=" * 60)

        # 模拟搜索结果
        mock_results = [
            {"idx": 1, "title": "Hans Arp - Wikipedia", "snippet": "Hans Arp (1886-1966) was a German-French sculptor, painter, and poet who was a founding member of Dada movement in Zurich."},
            {"idx": 2, "title": "Tina Turner Love Explosion Album", "snippet": "Love Explosion is the eighth studio album by Tina Turner, released in 1979. The cover features artwork."},
            {"idx": 3, "title": "Dada Art Movement History", "snippet": "Dada was an art movement of the European avant-garde in the early 20th century, started in Zurich, Switzerland."},
            {"idx": 4, "title": "Jean Arp Sculptures Collection", "snippet": "Jean Arp, also known as Hans Arp, created many plaster and bronze sculptures. His work appeared on various album covers."},
            {"idx": 5, "title": "Swiss Chocolate History", "snippet": "Switzerland is famous for its chocolate industry, with brands like Lindt and Toblerone."},
        ]

        test_query = "Hans Arp Swiss Dada artist plaster sculpture Tina Turner album death year"
        print(f"\n查询: {test_query}")
        print(f"\n原始搜索结果 ({len(mock_results)} 条):")
        for r in mock_results:
            print(f"  [{r['idx']}] {r['title']}")
            print(f"      {r['snippet'][:60]}...")

        # 运行 result reflection（单独函数）
        scored_results = await reflect_search_results(llm, test_query, mock_results)

        print(f"\n评分后排序结果:")
        for r in scored_results:
            sim = r.get("similarity_score", "N/A")
            overall = r.get("overall_score", "N/A")
            print(f"  [{r['idx']}] {r['title']}")
            print(f"      similarity={sim}, overall={overall}")

        # 版本2：full（reflect + rollout + result_reflection）
        state_full: DeepResearchState = {
            **state,
            "search_results_by_query": {
                test_queries[0]: [dict(x) for x in mock_results],
            },
        }
        node_v2 = make_query_optimize_node_full(llm, flash_llm=flash_llm)
        new_state_v2 = await node_v2(state_full)
        print("\n" + "=" * 60)
        print("v2 输出（full）:")
        print("=" * 60)
        print(f"queries: {len(new_state_v2.get('queries', []))}")
        print(f"reranked groups: {len(new_state_v2.get('reranked_search_results_by_query', {}))}")

    asyncio.run(_demo())
