# deepresearch/tools/fetch_tool.py
# -*- coding: utf-8 -*-
"""
抓取工具：给一个 URL，抓网页正文；如果是 PDF，抽取前几页文本。

"""
from __future__ import annotations

import asyncio
import os
import re
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ..schemas import Document


DEFAULT_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}


def _clean_text(text: str) -> str:
    # 简单清洗：去多余空白
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """将文本按窗口切块，供 query 定向抽取使用。"""
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: List[str] = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_size]
        if piece:
            chunks.append(piece)
        if i + chunk_size >= len(text):
            break
    return chunks


def _keyword_score(query: str, passage: str) -> int:
    """非常轻量的词命中打分，避免额外依赖。"""
    if not query or not passage:
        return 0
    terms = [t.strip().lower() for t in re.split(r"[\s,;，。！？:：()\[\]{}\"'`]+", query) if t.strip()]
    if not terms:
        return 0
    p = passage.lower()
    return sum(1 for t in terms if len(t) >= 2 and t in p)


def _extract_query_passages(
    text: str,
    query: Optional[str],
    top_k: int = 3,
    chunk_size: int = 800,
    overlap: int = 120,
) -> str:
    """按 query 抽取最相关段落；无 query 时返回原文。"""
    if not query or not query.strip() or not text:
        return text

    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return text

    scored = [(_keyword_score(query, c), idx, c) for idx, c in enumerate(chunks)]
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    picked = [c for s, _, c in scored[: max(1, top_k)] if s > 0]
    if not picked:
        return text

    merged = "\n\n".join(picked)
    return _clean_text(merged)


def _build_headers(url: str, user_agent: Optional[str] = None) -> dict:
    headers = dict(DEFAULT_BROWSER_HEADERS)
    if user_agent:
        headers["User-Agent"] = user_agent

    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        headers["Referer"] = origin
        headers["Origin"] = origin
    return headers


class SimpleFetcher:
    """
    简易网页抓取工具，支持 HTML 和 PDF。

    主要功能：
    - 抓取 HTML 网页并提取纯文本（去除 script/style 标签）。
    - 下载 PDF 文件并提取前几页文本。
    - 对提取的文本进行基本的清洗和长度截断。
    """
    def __init__(self, timeout_s: float = 20.0, max_chars: int = 12000):
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self.max_retries = max(1, int(os.getenv("FETCH_RETRIES", "2")))

    async def fetch(self, url: str, query: Optional[str] = None) -> Document:
        """
        抓取指定 URL 的网页内容。

        Args:
            url (str): 目标网页的 URL 地址。

        Returns:
            Document: 抓取结果对象。
                - url (str): 原始 URL。
                - title (Optional[str]): 网页标题（如果提取成功）。
                - content (str): 清洗后的正文文本（若超过 max_chars 会被截断）。
        
        Raises:
            httpx.HTTPStatusError: 如果 HTTP 请求返回 4xx/5xx 错误。
            httpx.RequestError: 如果网络请求发生错误（如超时、连接失败）。
        """
        timeout = httpx.Timeout(connect=min(10.0, self.timeout_s), read=self.timeout_s, write=self.timeout_s, pool=self.timeout_s)
        headers = _build_headers(url)

        last_exc: Optional[Exception] = None
        resp = None
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, http2=True) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if exc.response is not None and exc.response.status_code in {403, 429} and attempt < self.max_retries:
                        await asyncio.sleep(0.8 * attempt)
                        continue
                    raise
                except httpx.RequestError as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.8 * attempt)
                        continue
                    raise

            if resp is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(f"fetch failed without response: {url}")

            content_type = (resp.headers.get("content-type") or "").lower()

            # 1) PDF：用 pypdf 提取前几页文本
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                return await self._parse_pdf(url, resp.content)

            # 2) HTML：用 BeautifulSoup 提取纯文本
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            # 去掉脚本/样式等
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            title = None
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            text = soup.get_text("\n")
            text = _clean_text(text)
            if len(text) > self.max_chars:
                text = text[: self.max_chars] + "\n\n[内容截断]"

            text = _extract_query_passages(
                text,
                query=query,
                top_k=int(os.getenv("FETCH_QUERY_TOPK", "3")),
                chunk_size=int(os.getenv("FETCH_QUERY_CHUNK_SIZE", "800")),
                overlap=int(os.getenv("FETCH_QUERY_CHUNK_OVERLAP", "120")),
            )
            return Document(url=url, title=title, content=text)

    async def _parse_pdf(self, url: str, data: bytes) -> Document:
        import importlib

        PdfReader = None
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = importlib.import_module(module_name)
                PdfReader = getattr(module, "PdfReader", None)
                if PdfReader is not None:
                    break
            except Exception:
                continue
        if PdfReader is None:
            raise RuntimeError("未安装 PDF 解析依赖，请安装 pypdf（推荐）或 PyPDF2")

        reader = PdfReader(io_bytes := _BytesIO(data))
        texts = []
        max_pages = min(5, len(reader.pages))  # baseline：只取前 5 页，先够用
        for i in range(max_pages):
            page = reader.pages[i]
            t = page.extract_text() or ""
            texts.append(t)

        content = _clean_text("\n\n".join(texts))
        if len(content) > self.max_chars:
            content = content[: self.max_chars] + "\n\n[内容截断]"

        return Document(url=url, title="PDF Document", content=content)


class JinaReaderFetcher:
    """
    使用 Jina Reader 抓取网页正文。

    原理：请求 https://r.jina.ai/http://target-url
    """

    def __init__(
        self,
        timeout_s: float = 30.0,
        max_chars: int = 12000,
        base_url: str = "https://r.jina.ai/http://",
        api_key: Optional[str] = None,
        engine: str = "direct",
        return_format: str = "text",
        token_budget: int = 50000,
        extra_headers: Optional[dict] = None,
    ):
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.engine = engine
        self.return_format = return_format
        self.token_budget = token_budget
        self.extra_headers = extra_headers or {}

    async def fetch(self, url: str, query: Optional[str] = None) -> Document:
        target = url.strip()
        if target.startswith("http://"):
            tail = target[len("http://"):]
        elif target.startswith("https://"):
            tail = target[len("https://"):]
        else:
            tail = target

        reader_url = f"{self.base_url}{tail}"
        headers = {
            **_build_headers(url, user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )),
            "X-Engine": self.engine,
            "X-Return-Format": self.return_format,
            "X-Timeout": str(int(self.timeout_s)),
            "X-Token-Budget": str(int(self.token_budget)),
        }
        if self.api_key:
            # 若你的 Jina Reader 网关启用了鉴权，可通过该 header 透传
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
            resp = await client.get(reader_url, headers=headers)
            resp.raise_for_status()
            text = _clean_text(resp.text)

        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[内容截断]"

        text = _extract_query_passages(
            text,
            query=query,
            top_k=int(os.getenv("FETCH_QUERY_TOPK", "3")),
            chunk_size=int(os.getenv("FETCH_QUERY_CHUNK_SIZE", "800")),
            overlap=int(os.getenv("FETCH_QUERY_CHUNK_OVERLAP", "120")),
        )

        return Document(url=url, title=f"JinaReader: {url}", content=text)


class HybridFetcher:
    """
    迁移版抓取器：按顺序尝试多个后端，提升稳定性。
    默认顺序：Simple -> Jina(direct) -> Jina(browser)
    """

    def __init__(self, fetchers: List[object]):
        self.fetchers = fetchers

    async def fetch(self, url: str, query: Optional[str] = None) -> Document:
        last_exc: Optional[Exception] = None
        for f in self.fetchers:
            try:
                return await f.fetch(url, query=query)
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("HybridFetcher 未配置任何可用 fetcher")


def build_fetcher():
    """
    工厂方法：默认使用更稳的 HybridFetcher。

    切换到 Jina Reader：
    - FETCH_READER=jina  或
    - USE_JINA_READER=1
    可选：
    - JINA_READER_BASE_URL（默认 https://r.jina.ai/http://）
    - JINA_API_KEY（如网关需要鉴权）
    """
    timeout_s = float(os.getenv("FETCH_TIMEOUT_S", "20"))
    max_chars = int(os.getenv("FETCH_MAX_CHARS", "12000"))

    mode = os.getenv("FETCH_READER", "").strip().lower()
    fetch_mode = os.getenv("FETCH_MODE", "hybrid").strip().lower()  # simple | jina | hybrid
    use_jina = os.getenv("USE_JINA_READER", "0").strip().lower() in {"1", "true", "yes"}

    base_url = os.getenv("JINA_READER_BASE_URL", "https://r.jina.ai/http://").strip()
    api_key = os.getenv("JINA_API_KEY", "").strip() or None

    if fetch_mode == "jina" or mode == "jina" or use_jina:
        return JinaReaderFetcher(
            timeout_s=timeout_s,
            max_chars=max_chars,
            base_url=base_url,
            api_key=api_key,
            engine=os.getenv("JINA_ENGINE", "direct").strip() or "direct",
            return_format=os.getenv("JINA_RETURN_FORMAT", "text").strip() or "text",
            token_budget=int(os.getenv("JINA_TOKEN_BUDGET", "50000")),
        )

    if fetch_mode == "simple" or mode == "simple":
        return SimpleFetcher(timeout_s=timeout_s, max_chars=max_chars)

    if fetch_mode == "hybrid":
        simple = SimpleFetcher(timeout_s=timeout_s, max_chars=max_chars)
        jina_direct = JinaReaderFetcher(
            timeout_s=max(timeout_s, 20.0),
            max_chars=max_chars,
            base_url=base_url,
            api_key=api_key,
            engine="direct",
            return_format="text",
            token_budget=int(os.getenv("JINA_TOKEN_BUDGET", "50000")),
        )
        jina_browser = JinaReaderFetcher(
            timeout_s=max(timeout_s, 20.0),
            max_chars=max_chars,
            base_url=base_url,
            api_key=api_key,
            engine="browser",
            return_format="text",
            token_budget=int(os.getenv("JINA_BROWSER_TOKEN_BUDGET", "80000")),
        )
        return HybridFetcher([simple, jina_direct, jina_browser])

    return HybridFetcher([
        SimpleFetcher(timeout_s=timeout_s, max_chars=max_chars),
        JinaReaderFetcher(
            timeout_s=max(timeout_s, 20.0),
            max_chars=max_chars,
            base_url=base_url,
            api_key=api_key,
            engine="direct",
            return_format="text",
            token_budget=int(os.getenv("JINA_TOKEN_BUDGET", "50000")),
        ),
        JinaReaderFetcher(
            timeout_s=max(timeout_s, 20.0),
            max_chars=max_chars,
            base_url=base_url,
            api_key=api_key,
            engine="browser",
            return_format="text",
            token_budget=int(os.getenv("JINA_BROWSER_TOKEN_BUDGET", "80000")),
        ),
    ])


class _BytesIO:
    """避免引入 io.BytesIO 让新手困惑（其实就是 BytesIO 的最小替代）"""
    def __init__(self, data: bytes):
        import io
        self._bio = io.BytesIO(data)

    def read(self, *args, **kwargs):
        return self._bio.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._bio.seek(*args, **kwargs)

    def tell(self):
        return self._bio.tell()


if __name__ == "__main__":
    async def _demo() -> None:
        fetcher = SimpleFetcher()
        test_url = "https://www.python.org/"
        try:
            doc = await fetcher.fetch(test_url)
        except Exception as exc:  # 快速验证用，详细错误交给调用方处理
            print(f"fetch failed: {exc}")
            return

        print(f"fetched url: {doc.url}")
        print(f"title: {doc.title}")
        preview = (doc.content or "")[:200].replace("\n", " ")
        print(f"content preview (200 chars): {preview}")

    asyncio.run(_demo())