# deepresearch/nodes/retrieve.py
# -*- coding: utf-8 -*-
"""节点：retrieve

执行 search + fetch。
新版支持循环式研究：
- 增量抓取（保留历史 documents）
- query_history 去重
- 跨轮避免重复 URL
"""
from __future__ import annotations

import asyncio
from typing import Callable, List, Set, Any, Tuple

from langchain_core.messages import AIMessage

from ..schemas import Document, SearchResult
from ..state import DeepResearchState


def _normalize_query(q: Any) -> str:
    """兼容未来 query pack：{"q": "...", ...}"""
    if isinstance(q, str):
        return q.strip()
    if isinstance(q, dict):
        return str(q.get("q", "")).strip()
    return str(q).strip()


def make_retrieve_node(searcher, fetcher, max_docs: int = 8, per_query_results: int = 3):
    async def retrieve(state: DeepResearchState) -> DeepResearchState:
        print("\n============ retrieve 阶段 ============")
        raw_queries = state.get("queries", []) or []
        queries = [_normalize_query(q) for q in raw_queries if _normalize_query(q)]
        existing_docs: List[Document] = state.get("documents", []) or []
        existing_urls = {d.url for d in existing_docs if getattr(d, "url", None)}
        print(f"[retrieve] raw_queries={len(raw_queries)}, normalized_queries={len(queries)}")
        for i, q in enumerate(queries, start=1):
            print(f"[retrieve] query_{i}: {q}")
        if not queries:
            return {"documents": [], "messages": [AIMessage(content="[retrieve] 没有 queries，跳过抓取。")]}

        print(f"[retrieve] max_docs={max_docs}, per_query_results={per_query_results}")

        seen_urls: Set[str] = set(existing_urls)
        candidate_items: List[Tuple[str, str]] = []

        # 1) 先 search：收集候选 URL（去重 + 限额）
        for query in queries:
            if len(candidate_items) >= max_docs:
                break
            try:
                results: List[SearchResult] = await searcher.search(query)
                print(f"[retrieve] search_results for '{query}': {len(results)}")
            except Exception:
                print(f"[retrieve] search_error for query: {query}")
                continue

            for r in results[:per_query_results]:
                print(f"[retrieve] result: title={getattr(r, 'title', '')} url={getattr(r, 'url', '')} snippet={getattr(r, 'snippet', '')}")
                if len(candidate_items) >= max_docs:
                    break
                if not r.url or r.url in seen_urls:
                    continue
                seen_urls.add(r.url)
                candidate_items.append((r.url, query))

        if not candidate_items:
            return {"documents": [], "messages": [AIMessage(content="[retrieve] 搜索无结果或全部被去重/过滤。")]}
        print(f"[retrieve] candidate_urls={len(candidate_items)}")
        for i, (u, q) in enumerate(candidate_items, start=1):
            print(f"[retrieve] candidate_{i}: {u} | from_query={q}")

        # 2) 全部并发 fetch
        async def _fetch_one(url: str, query: str):
            try:
                try:
                    doc: Document = await fetcher.fetch(url, query=query)
                except TypeError:
                    # 兼容旧 fetcher（只接受 url）
                    doc = await fetcher.fetch(url)
                return doc
            except Exception:
                return None

        new_docs: List[Document] = []
        tasks = [asyncio.create_task(_fetch_one(u, q)) for (u, q) in candidate_items]
        for fut in asyncio.as_completed(tasks):
            doc = await fut
            if doc is not None:
                if doc.url in existing_urls:
                    continue
                new_docs.append(doc)
                print(f"[retrieve] fetched_doc: url={doc.url} title={doc.title} content_len={len(doc.content or '')}")
                print("[retrieve] --- doc_content begin ---")
                print(doc.content)
                print("[retrieve] --- doc_content end ---")
            if len(new_docs) >= max_docs:
                break

        merged_docs = existing_docs + new_docs
        query_history = state.get("query_history", []) or []
        query_history.extend(queries)
        # 保序去重
        dedup_history = list(dict.fromkeys([q for q in query_history if q]))

        msg = AIMessage(
            content=(
                f"[retrieve] 本轮查询 {len(queries)} 条，候选 {len(candidate_items)}，"
                f"新增文档 {len(new_docs)}，累计文档 {len(merged_docs)}。"
            )
        )
        print(f"[retrieve] fetched_new_docs={len(new_docs)}, total_docs={len(merged_docs)}")
        return {"documents": merged_docs, "query_history": dedup_history, "messages": [msg]}

    return retrieve


if __name__ == "__main__":
    async def _demo() -> None:
        class DummyResult:
            def __init__(self, title: str, url: str, snippet: str = None):
                self.title = title
                self.url = url
                self.snippet = snippet

        class DummySearcher:
            async def search(self, query: str):
                # 模拟返回 2 条结果，包含重复 URL 用于测试去重
                return [
                    DummyResult(title=f"Result for {query} #1", url=f"https://example.com/{hash(query)%1000}", snippet="stub"),
                    DummyResult(title=f"Result for {query} #2", url=f"https://example.com/{hash(query)%1000}", snippet="stub"),
                ]

        class DummyFetcher:
            async def fetch(self, url: str) -> Document:
                return Document(url=url, title="Dummy", content=f"content for {url}")

        searcher = DummySearcher()
        fetcher = DummyFetcher()
        node = make_retrieve_node(searcher, fetcher)

        # 使用前一步 parse_claims 生成的 claims 手工构造 queries
        queries = [
            "艺术家A 25岁 毕业 学校B",
            "学校B 中国近代作家C 命名",
            "作家C 出生 中国南方城市",
            "艺术家A 2012 拍卖 超过500万",
            "艺术家A 2021 拍卖 超过2000万",
        ]

        state: DeepResearchState = {
            "queries": queries,
        }

        new_state = await node(state)
        print(f"documents: {len(new_state.get('documents', []))}")
        for i, doc in enumerate(new_state.get("documents", []), start=1):
            preview = doc.content[:60].replace("\n", " ")
            print(f"[{i}] {doc.url} | {preview}")
        for msg in new_state.get("messages", []):
            print(msg.content)

    asyncio.run(_demo())