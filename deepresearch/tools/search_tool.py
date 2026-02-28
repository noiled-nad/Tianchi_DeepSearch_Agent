# deepresearch/tools/search_tool.py
# -*- coding: utf-8 -*-
"""
搜索工具（多源版）：
- 支持 Serper / 阿里云 IQS / DuckDuckGo / Wikipedia（兼容 SerpApi）
- 支持多源并发聚合、去重与限额
- 默认可通过环境变量开启“多源检索”

注意：
- 你现在报错是“Network is unreachable”，说明运行环境可能没出网。
  若完全不能访问外网，那么 SerpApi 也同样无法使用。
"""

from __future__ import annotations

import os
import asyncio
from typing import List, Optional, Protocol
from urllib.parse import urlparse

import httpx

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from ..schemas import SearchResult
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from deepresearch.schemas import SearchResult


def _getenv(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else ""


def _getenv_int(name: str, default: int) -> int:
    raw = _getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


def _getenv_float(name: str, default: float) -> float:
    raw = _getenv(name, str(default)).strip()
    try:
        return float(raw)
    except Exception:
        return default


class AsyncSearcher(Protocol):
    name: str

    async def search(self, query: str) -> List[SearchResult]:
        ...


class SerpApiSearcher:
    """
    通过 SerpApi 调用搜索引擎，返回结构化 JSON 结果。
    官方支持 /search.json 接口。:contentReference[oaicite:3]{index=3}
    """

    def __init__(self, api_key: str, engine: str = "google", max_results: int = 5, timeout_s: float = 20.0):
        self.name = f"serpapi:{engine}"
        self.api_key = api_key
        self.engine = engine
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None

    async def search(self, query: str) -> List[SearchResult]:
        self.last_error = None
        # SerpApi 的 endpoint：/search.json
        # 你可以用 engine=google / engine=baidu 等。:contentReference[oaicite:4]{index=4}
        url = "https://serpapi.com/search.json"
        params = {
            "engine": self.engine,
            "api_key": self.api_key,
        }
        if self.engine == "yahoo":
            params["p"] = query
        else:
            params["q"] = query
        if self.engine == "bing":
            params["count"] = self.max_results
        elif self.engine == "baidu":
            params["rn"] = self.max_results
        else:
            params["num"] = self.max_results

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            # 断网、超时、配额错误等都不要让流程崩，直接空结果
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
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


class SerperSearcher:
    """
    Serper.dev 搜索（Google）。
    """

    def __init__(self, api_key: str, max_results: int = 5, timeout_s: float = 20.0, safe: str = "active"):
        self.name = "serper"
        self.api_key = api_key
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.safe = (safe or "active").strip().lower()
        self.last_error: Optional[str] = None

    async def search(self, query: str) -> List[SearchResult]:
        self.last_error = None
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": self.max_results,
            "safe": self.safe,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            return []

        items = data.get("organic") or []
        results: List[SearchResult] = []
        for item in items:
            title = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            snippet = item.get("snippet")
            if title and link:
                results.append(SearchResult(title=title, url=link, snippet=snippet))
        return results[: self.max_results]


class AliyunIQSSearcher:
    """
    阿里云 IQS unified search。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://cloud-iqs.aliyuncs.com",
        engine_type: str = "Generic",
        auth_mode: str = "bearer",
        max_results: int = 5,
        timeout_s: float = 20.0,
    ):
        self.name = "aliyun_iqs"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.engine_type = engine_type
        self.auth_mode = auth_mode.lower().strip() or "bearer"
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None

    async def search(self, query: str) -> List[SearchResult]:
        self.last_error = None
        url = f"{self.base_url}/search/unified"
        headers = {"Content-Type": "application/json"}
        if self.auth_mode in ("x-api-key", "x_api_key", "xapikey"):
            headers["X-API-Key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "query": query,
            "engineType": self.engine_type,
            "contents": {
                "mainText": False,
                "markdownText": False,
                "summary": False,
                "rerankScore": True,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            return []

        items = data.get("pageItems") or (data.get("result") or {}).get("pageItems") or []
        results: List[SearchResult] = []
        for item in items:
            title = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            snippet = item.get("snippet") or item.get("summary") or item.get("mainText")
            if title and link:
                results.append(SearchResult(title=title, url=link, snippet=snippet))
        return results[: self.max_results]


class BochaSearcher:
    """
    Bocha Web Search API。
    """

    def __init__(self, api_key: str, max_results: int = 5, timeout_s: float = 20.0):
        self.name = "bocha"
        self.api_key = api_key
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None

    async def search(self, query: str) -> List[SearchResult]:
        self.last_error = None
        url = "https://api.bochaai.com/v1/web-search"
        payload = {
            "query": query,
            "summary": True,
            "count": self.max_results,
            "page": 1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            return []

        if data.get("code") != 200:
            return []
        items = ((data.get("data") or {}).get("webPages") or {}).get("value") or []

        results: List[SearchResult] = []
        for item in items:
            title = (item.get("name") or "").strip()
            link = (item.get("url") or "").strip()
            snippet = item.get("snippet")
            if title and link:
                results.append(SearchResult(title=title, url=link, snippet=snippet))
        return results[: self.max_results]


class DuckDuckGoSearcher:
    """
    备用：DuckDuckGo（如果 SerpApi 没配置 key）
    """

    def __init__(self, max_results: int = 5):
        self.name = "duckduckgo"
        self.max_results = max_results
        self.last_error: Optional[str] = None

    def _search_sync(self, query: str) -> List[SearchResult]:
        self.last_error = None
        try:
            from ddgs import DDGS
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
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
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            return []

        return results

    async def search(self, query: str) -> List[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query)


class WikipediaSearcher:
    """
    轻量 Wikipedia 搜索：适合补充定义类/实体类信息。
    """

    def __init__(self, max_results: int = 3, timeout_s: float = 10.0):
        self.name = "wikipedia"
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None

    async def search(self, query: str) -> List[SearchResult]:
        self.last_error = None
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": self.max_results,
            "utf8": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            return []

        items = (data.get("query") or {}).get("search") or []
        results: List[SearchResult] = []
        for item in items:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            snippet = (item.get("snippet") or "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            results.append(SearchResult(title=title, url=page_url, snippet=snippet or None))
        return results


class MultiSourceSearcher:
    """
    多源搜索聚合器：
    - 并发调用各搜索源
    - URL 去重
    - 交错合并（提升来源多样性）
    """

    def __init__(self, searchers: List[AsyncSearcher], max_results: int = 8):
        self.name = "multi_source"
        self.searchers = searchers
        self.max_results = max_results
        self.last_source_counts: dict[str, int] = {}
        self.last_source_errors: dict[str, str] = {}

    @staticmethod
    def _normalize_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return ""
        parsed = urlparse(u)
        if not parsed.scheme:
            return u
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return f"{parsed.scheme.lower()}://{netloc}{parsed.path}".rstrip("/")

    async def search(self, query: str) -> List[SearchResult]:
        if not self.searchers:
            return []

        tasks = [s.search(query) for s in self.searchers]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        self.last_source_counts = {}
        self.last_source_errors = {}

        buckets: List[List[SearchResult]] = []
        for i, item in enumerate(raw):
            src_name = getattr(self.searchers[i], "name", f"source_{i+1}")
            if isinstance(item, Exception):
                self.last_source_counts[src_name] = 0
                self.last_source_errors[src_name] = str(item)
                buckets.append([])
                continue
            results = item if isinstance(item, list) else []
            self.last_source_counts[src_name] = len(results)
            src_error = getattr(self.searchers[i], "last_error", None)
            if src_error:
                self.last_source_errors[src_name] = src_error
            tagged: List[SearchResult] = []
            for r in results:
                snippet = r.snippet or ""
                if not snippet.startswith("["):
                    snippet = f"[{src_name}] {snippet}".strip()
                tagged.append(SearchResult(title=r.title, url=r.url, snippet=snippet or None))
            buckets.append(tagged)

        merged: List[SearchResult] = []
        seen: set[str] = set()
        cursor = 0
        while len(merged) < self.max_results:
            progressed = False
            for bucket in buckets:
                if cursor >= len(bucket):
                    continue
                r = bucket[cursor]
                norm = self._normalize_url(r.url)
                if norm and norm not in seen:
                    seen.add(norm)
                    merged.append(r)
                    progressed = True
                    if len(merged) >= self.max_results:
                        break
            if not progressed:
                break
            cursor += 1

        return merged


def _parse_sources_from_env(default: str) -> List[str]:
    raw = _getenv("SEARCH_SOURCES", default)
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return items or [x.strip().lower() for x in default.split(",") if x.strip()]


def _build_from_source_spec(
    spec: str,
    *,
    serp_key: str,
    bocha_key: str,
    per_source_results: int,
    timeout_s: float,
) -> Optional[AsyncSearcher]:
    spec = spec.strip().lower()
    if not spec:
        return None

    serper_key = _getenv("SERPER_API_KEY", "").strip()
    iqs_key = _getenv("IQS_API_KEY", "").strip()

    if spec in ("serper", "google"):
        if not serper_key:
            return None
        serper_safe = _getenv("SERPER_SAFE", "active").strip().lower() or "active"
        return SerperSearcher(
            api_key=serper_key,
            max_results=per_source_results,
            timeout_s=timeout_s,
            safe=serper_safe,
        )

    if spec in ("iqs", "aliyun_iqs", "aliyun-iqs"):
        if not iqs_key:
            return None
        iqs_base = _getenv("IQS_BASE_URL", "https://cloud-iqs.aliyuncs.com").strip() or "https://cloud-iqs.aliyuncs.com"
        iqs_engine_type = _getenv("IQS_ENGINE_TYPE", "Generic").strip() or "Generic"
        iqs_auth_mode = _getenv("IQS_AUTH_MODE", "bearer").strip() or "bearer"
        return AliyunIQSSearcher(
            api_key=iqs_key,
            base_url=iqs_base,
            engine_type=iqs_engine_type,
            auth_mode=iqs_auth_mode,
            max_results=per_source_results,
            timeout_s=timeout_s,
        )

    serp_aliases = {"google", "bing", "baidu", "yahoo"}
    if spec in serp_aliases:
        if not serp_key:
            return None
        return SerpApiSearcher(api_key=serp_key, engine=spec, max_results=per_source_results, timeout_s=timeout_s)

    if spec.startswith("serpapi"):
        if not serp_key:
            return None
        if ":" in spec:
            _, engine = spec.split(":", 1)
            engine = engine.strip() or "google"
        else:
            engine = _getenv("SERPAPI_ENGINE", "google").strip() or "google"
        return SerpApiSearcher(api_key=serp_key, engine=engine, max_results=per_source_results, timeout_s=timeout_s)

    if spec in ("duckduckgo", "ddg"):
        return DuckDuckGoSearcher(max_results=per_source_results)

    if spec in ("wikipedia", "wiki"):
        return WikipediaSearcher(max_results=min(per_source_results, 10), timeout_s=min(timeout_s, 20.0))

    if spec == "bocha":
        if not bocha_key:
            return None
        return BochaSearcher(api_key=bocha_key, max_results=per_source_results, timeout_s=timeout_s)

    return None


def build_searcher():
    """
    工厂方法：根据环境变量组装搜索器。

    可选环境变量：
    - SEARCH_SOURCES: 逗号分隔，如 "serper,iqs,duckduckgo,wikipedia"
    - SEARCH_MAX_RESULTS: 聚合后的总结果数（默认 8）
    - SEARCH_PER_SOURCE_RESULTS: 每个搜索源返回条数（默认 5）
    - SEARCH_TIMEOUT_S: 网络请求超时（默认 20 秒）
    - SERPER_API_KEY: Serper key
    - SERPER_SAFE: Serper 安全搜索级别（默认 active，可选 off）
    - IQS_API_KEY: 阿里云 IQS key
    - IQS_BASE_URL: IQS 服务地址（默认 https://cloud-iqs.aliyuncs.com）
    - IQS_ENGINE_TYPE: IQS 引擎类型（默认 Generic）
    - IQS_AUTH_MODE: bearer 或 x-api-key（默认 bearer）

    兼容旧配置：
    - SERPAPI_API_KEY: SerpApi key
    - SERPAPI_ENGINE: 单独使用 serpapi 时的默认引擎（默认 google）
    - BOCHA_API_KEY: Bocha key（启用 bocha 源时需要）

    SEARCH_SOURCES 示例：
    - serper,iqs,duckduckgo,wikipedia
    - serper,duckduckgo,wikipedia
    - iqs,duckduckgo

    旧示例（仍可用）：
    - serpapi:google,duckduckgo,wikipedia
    - google,bing,duckduckgo
    - bocha,duckduckgo,wiki
    """
    serp_key = _getenv("SERPAPI_API_KEY", "").strip()
    bocha_key = _getenv("BOCHA_API_KEY", "").strip()
    per_source_results = _getenv_int("SEARCH_PER_SOURCE_RESULTS", _getenv_int("SERPAPI_MAX_RESULTS", 5))
    total_results = _getenv_int("SEARCH_MAX_RESULTS", 8)
    timeout_s = _getenv_float("SEARCH_TIMEOUT_S", 20.0)

    default_sources = "serper,iqs,duckduckgo,wikipedia"
    source_specs = _parse_sources_from_env(default_sources)

    searchers: List[AsyncSearcher] = []
    for spec in source_specs:
        s = _build_from_source_spec(
            spec,
            serp_key=serp_key,
            bocha_key=bocha_key,
            per_source_results=per_source_results,
            timeout_s=timeout_s,
        )
        if s is not None:
            searchers.append(s)

    if not searchers:
        return DuckDuckGoSearcher(max_results=per_source_results)
    if len(searchers) == 1:
        return searchers[0]
    min_total = per_source_results * len(searchers)
    if total_results < min_total:
        total_results = min_total
    return MultiSourceSearcher(searchers=searchers, max_results=total_results)


if __name__ == "__main__":
    async def _demo() -> None:
        if load_dotenv is not None:
            load_dotenv()
        query = _getenv("SEARCH_DEMO_QUERY", "OpenAI GPT-5 latest updates")
        searcher = build_searcher()

        print("=" * 12, "search_tool demo", "=" * 12)
        print(f"searcher_type: {type(searcher).__name__}")
        print(f"searcher_name: {getattr(searcher, 'name', 'unknown')}")
        print(f"query: {query}")

        if isinstance(searcher, MultiSourceSearcher):
            source_names = [getattr(s, "name", type(s).__name__) for s in searcher.searchers]
            print(f"is_multi_source: True")
            print(f"sources({len(source_names)}): {source_names}")
            print(f"merged_limit: {searcher.max_results}")
        else:
            print("is_multi_source: False")

        results = await searcher.search(query)
        print(f"results_count: {len(results)}")
        if isinstance(searcher, MultiSourceSearcher):
            print(f"source_raw_counts: {searcher.last_source_counts}")
            if searcher.last_source_errors:
                print(f"source_errors: {searcher.last_source_errors}")
        for i, r in enumerate(results, start=1):
            snippet = (r.snippet or "").replace("\n", " ")[:160]
            print(f"[{i}] {r.title}")
            print(f"    url: {r.url}")
            if snippet:
                print(f"    snippet: {snippet}")

    asyncio.run(_demo())
