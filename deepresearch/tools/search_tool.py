# deepresearch/tools/search_tool.py
# -*- coding: utf-8 -*-
"""
搜索工具（最小可用版）：
- 优先使用 SerpApi（你有 key）
- 如果没配置 SerpApi key，则退回 DuckDuckGo（如果你环境能出网）

注意：
- 你现在报错是“Network is unreachable”，说明运行环境可能没出网。
  若完全不能访问外网，那么 SerpApi 也同样无法使用。
"""

from __future__ import annotations

import os
import asyncio
from typing import List, Optional

import httpx

from ..schemas import SearchResult


def _getenv(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else ""


class SerpApiSearcher:
    """
    通过 SerpApi 调用搜索引擎，返回结构化 JSON 结果。
    官方支持 /search.json 接口。:contentReference[oaicite:3]{index=3}
    """

    def __init__(self, api_key: str, engine: str = "google", max_results: int = 5, timeout_s: float = 20.0):
        self.api_key = api_key
        self.engine = engine
        self.max_results = max_results
        self.timeout_s = timeout_s

    async def search(self, query: str) -> List[SearchResult]:
        # SerpApi 的 endpoint：/search.json
        # 你可以用 engine=google / engine=baidu 等。:contentReference[oaicite:4]{index=4}
        url = "https://serpapi.com/search.json"
        params = {
            "engine": self.engine,
            "q": query,
            "api_key": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            # 断网、超时、配额错误等都不要让流程崩，直接空结果
            return []

        # SerpApi 返回里最常用的是 organic_results
        organic = data.get("organic_results") or []
        results: List[SearchResult] = []
        for r in organic:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or None)
            if title and link:
                results.append(SearchResult(title=title, url=link, snippet=snippet))

        return results[: self.max_results]


class DuckDuckGoSearcher:
    """
    备用：DuckDuckGo（如果 SerpApi 没配置 key）
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def _search_sync(self, query: str) -> List[SearchResult]:
        try:
            from ddgs import DDGS
        except Exception:
            return []

        results: List[SearchResult] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    url = r.get("href") or ""
                    title = r.get("title") or ""
                    snippet = r.get("body")
                    if url and title:
                        results.append(SearchResult(title=title, url=url, snippet=snippet))
        except Exception:
            return []

        return results

    async def search(self, query: str) -> List[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query)


def build_searcher():
    """
    工厂方法：根据环境变量选择搜索器。
    """
    serp_key = _getenv("SERPAPI_API_KEY", "")
    if serp_key.strip():
        engine = _getenv("SERPAPI_ENGINE", "google").strip() or "google"
        max_results = int(_getenv("SERPAPI_MAX_RESULTS", "5"))
        return SerpApiSearcher(api_key=serp_key, engine=engine, max_results=max_results)

    # 没配 SerpApi key 才 fallback
    return DuckDuckGoSearcher(max_results=int(_getenv("SERPAPI_MAX_RESULTS", "5")))
