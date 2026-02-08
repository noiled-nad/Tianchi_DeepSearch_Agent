# deepresearch/nodes/retrieve.py
# -*- coding: utf-8 -*-
"""
节点3：retrieve
目标：执行 search + fetch，把网页正文保存到 documents。
"""
from __future__ import annotations

import asyncio
from typing import Callable, List, Set

from langchain_core.messages import AIMessage

from ..schemas import Document
from ..state import DeepResearchState


def make_retrieve_node(searcher, fetcher):
    async def retrieve(state: DeepResearchState) -> DeepResearchState:
        queries = state.get("queries",[])
        documents = []
        seen_urls = set()## 去重
        MAX_DOCS = 8     ## 设置最大的检索文档数
        for query in queries:
            if len(documents) >= MAX_DOCS: break

            results = await searcher.search(query)
            for  result in results[:3]:
                if len(documents) >= MAX_DOCS: break
                # 剪枝 4: 去重
                if result.url in seen_urls: continue
                seen_urls.add(result.url)
            
            try:
                doc = await fetcher.fetch(result.url)
                documents.append(doc)
            except Exception:
                continue
        return  {
            "documents":documents,
            "messages": [AIMessage(content=f"抓取了 {len(documents)} 篇")]
        }
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