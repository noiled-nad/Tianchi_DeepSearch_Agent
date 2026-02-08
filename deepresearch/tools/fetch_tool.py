# deepresearch/tools/fetch_tool.py
# -*- coding: utf-8 -*-
"""
抓取工具：给一个 URL，抓网页正文；如果是 PDF，抽取前几页文本。

"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ..schemas import Document


def _clean_text(text: str) -> str:
    # 简单清洗：去多余空白
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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

    async def fetch(self, url: str) -> Document:
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
        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "deepresearch-bot/0.1"})
            resp.raise_for_status()

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

            return Document(url=url, title=title, content=text)

    async def _parse_pdf(self, url: str, data: bytes) -> Document:
        from pypdf import PdfReader  # 延迟 import

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