"""deepresearch/tools/compress.py

文档压缩工具：用 flash LLM 从单个文档中提取与当前子任务相关的有用信息。
配合 fetch 构成原子级流水线：谁先抓完，谁就立刻进入 LLM 压缩。
网络 I/O 与 LLM I/O 完美交叠，互不阻塞。
"""

from __future__ import annotations

import asyncio
from typing import Optional


COMPRESS_PROMPT = """你是信息提取专家。请根据问题与当前子任务，从以下文档中提取所有相关有用信息。

原始问题：{question}
子任务：{subtask_title}
子任务目标：{subtask_reason}

文档内容：
{doc_content}

规则：
1) 只提取与子任务直接相关的事实、数据、名称、日期等关键信息
2) 保留原文措辞，不要改写或推测
3) 如果文档与子任务完全无关，只输出"无关"
4) 用简洁的要点列表输出，每条一行"""


async def compress_doc(
    flash_llm,
    question: str,
    subtask: dict,
    doc,
    max_doc_chars: int = 2000,
    timeout_s: float = 30.0,
) -> Optional[str]:
    """
    用 flash LLM 压缩单个文档，提取与子任务相关的信息。

    Args:
        flash_llm: flash LLM 实例
        question: 原始问题
        subtask: 子任务字典 (含 title, reason)
        doc: Document 对象
        max_doc_chars: 文档内容最大字符数
        timeout_s: LLM 调用超时秒数

    Returns:
        压缩后的文本片段（含来源标题和 URL），无关文档返回 None。
    """
    content = doc.content or ""
    if not content.strip():
        return None

    if len(content) > max_doc_chars:
        content = content[:max_doc_chars]

    prompt = COMPRESS_PROMPT.format(
        question=question,
        subtask_title=subtask.get("title", ""),
        subtask_reason=subtask.get("reason", ""),
        doc_content=content,
    )

    try:
        resp = await asyncio.wait_for(
            flash_llm.ainvoke(prompt), timeout=timeout_s
        )
        result = str(resp.content).strip()
        if not result or result == "无关":
            return None
        title = (doc.title or "").strip()
        url = doc.url or ""
        return f"{title}\nURL: {url}\n{result}"
    except asyncio.TimeoutError:
        print(f"    [compress] TIMEOUT({timeout_s}s) url='{(doc.url or '')[:80]}'")
        return None
    except Exception as e:
        print(f"    [compress] FAILED: {type(e).__name__}: {e}")
        return None
