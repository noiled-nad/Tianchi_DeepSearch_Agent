# deepresearch/nodes/extract_notes.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List
from ..state import DeepResearchState
from ..schemas import Claim, Document, Evidence
from langchain_core.messages import AIMessage
"""
把 documents 变成 可引用、可绑定 claim 的 Evidence（quote + url + claim_id）

"""

def _safe_json_list(text: str):
    # 1) 先尝试整体就是 JSON 数组
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        return json.loads(t)

    # 2) 兼容 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S | re.I)
    if fence:
        return json.loads(fence.group(1))

    # 3) 非贪婪扫描所有数组片段，挑出“像 Evidence 列表”的那个
    candidates = re.finditer(r"\[[\s\S]*?\]", text)
    for m in candidates:
        chunk = m.group(0)
        try:
            arr = json.loads(chunk)
        except Exception:
            continue
        if isinstance(arr, list):
            if not arr:
                return arr
            if isinstance(arr[0], dict) and ("claim_id" in arr[0] or "url" in arr[0] or "quote" in arr[0]):
                return arr

    raise ValueError("no valid evidence json array found")

def _compact_docs(docs: List[Document], max_chars_each: int = 1200) -> str:
    """
    每个文档格式为：
    [D1] 文档标题
    URL: https://example.com
    文档内容...
    """
    chunks = []
    for i, d in enumerate(docs, start=1):
        content = d.content or ""
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "\n[截断]"
        chunks.append(f"[D{i}] {d.title or ''}\nURL: {d.url}\n{content}")
    return "\n\n".join(chunks) if chunks else "（无文档）"


def make_extract_notes_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def extract_notes(state: DeepResearchState) -> DeepResearchState:
        print("\n============ extract_notes 阶段 ============")
        ## 根据已有的docs 和claims，解析可用的evidence
        claims: List[Claim] = state.get("claims", [])
        docs: List[Document] = state.get("documents", [])   
        print(f"[extract_notes] claims={len(claims)}, documents={len(docs)}")
        # 只给 must + 高价值类型，避免 token 爆
        type_rank = {"disamb": 0, "anchor": 1, "output": 2, "bridge": 3, "other": 9}
        focus = sorted(
            [c for c in claims if c.must or c.claim_type in ("disamb", "anchor", "output")],
            key=lambda c: type_rank.get(c.claim_type, 9),
        )[:12]  ## 只取 must的
        print(f"[extract_notes] focus_claims={len(focus)}")

        claims_text = "\n".join(
            f"- {c.id} ({c.claim_type}, must={c.must}): S={c.S.name} P={c.P} O={(c.O.name if c.O else '∅')} Q={c.Q.model_dump()} desc={c.description}"
            for c in focus
        )
        prompt = (
            "你是证据抽取器。任务：从【文档】中抽取可引用证据片段 Evidence，并绑定到最相关的 claim_id。\n"
            "必须严格输出 JSON 数组，每个元素结构：\n"
            "{claim_id, url, title, quote, snippet(optional), relevance(0~1), stance(supports/refutes/neutral)}\n"
            "硬约束：\n"
            "1) quote 必须是文档里的原文短摘录（<=300字符），不要自己改写。\n"
            "2) claim_id 必须来自下面的 claims。\n"
            "3) 每条 evidence 只绑定 1 个 claim_id。\n"
            "4) 若文档无法支持任何 claim，就输出空数组 []。\n\n"
            f"Claims:\n{claims_text}\n\n"
            f"Documents:\n{_compact_docs(docs)}\n"
        )
        print("[extract_notes] --- prompt begin ---")
        print(prompt)
        print("[extract_notes] --- prompt end ---")

        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        print(f"[extract_notes] raw_len={len(raw)}")
        print(f"[extract_notes] raw_response={raw}")

        try:
            arr = _safe_json_list(raw)
            evs = [Evidence(**e) for e in arr]
        except Exception as exc:
            preview = raw[:300].replace("\n", "\\n")
            print(f"[extract_notes] parse_error={exc}")
            print(f"[extract_notes] raw_preview={preview}")
            evs = []

        msg = AIMessage(content=f"[extract_notes] 抽取证据 {len(evs)} 条。")
        print(f"[extract_notes] evidences={len(evs)}")
        for i, e in enumerate(evs, start=1):
            print(f"[extract_notes] ev{i}: claim_id={e.claim_id}, url={e.url}, title={e.title}, relevance={e.relevance}, stance={e.stance}")
            print(f"[extract_notes] ev{i}_quote={e.quote}")
        return {"notes": evs, "messages": [msg]}

    return extract_notes


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from dotenv import load_dotenv
    from deepresearch.schemas import Claim, Document, Evidence, EntityRef, Qualifier
    from deepresearch.config import create_llm

    load_dotenv()

    async def _demo():
        # Create sample claims
        claims = [
            Claim(
                id="claim_1",
                must=True,
                claim_type="bridge",
                S=EntityRef(name="艺术家A"),
                P="毕业于",
                O=EntityRef(name="学校B"),
                Q=Qualifier(),
                description="艺术家A在25岁时从学校B毕业。"
            ),
            Claim(
                id="claim_2",
                must=True,
                claim_type="bridge",
                S=EntityRef(name="学校B"),
                P="以命名",
                O=EntityRef(name="作家C"),
                Q=Qualifier(),
                description="学校B以中国近代作家C的名字命名。"
            ),
            Claim(
                id="claim_3",
                must=False,
                claim_type="disamb",
                S=EntityRef(name="作家C"),
                P="出生于",
                O=EntityRef(name="南方城市"),
                Q=Qualifier(),
                description="作家C出生于中国南方城市。"
            )
        ]

        # Create sample documents
        docs = [
            Document(
                title="艺术家传记",
                url="https://example.com/artist",
                content="艺术家A是一位著名画家。他于25岁毕业于学校B，并在之后的作品中取得了巨大成功。"
            ),
            Document(
                title="学校历史",
                url="https://example.com/school",
                content="学校B成立于19世纪，以中国近代作家C的名字命名。该校培养了许多优秀艺术家。"
            )
        ]

        # Initial state
        state: DeepResearchState = {
            "claims": claims,
            "documents": docs
        }

        # Create and run node with real LLM
        llm = create_llm()
        node = make_extract_notes_node(llm)
        new_state = await node(state)

        # Print results
        print("=== 输入状态 ===")
        print(f"Claims: {len(claims)} 条")
        for c in claims:
            print(f"  - {c.id}: {c.description}")
        print(f"Documents: {len(docs)} 条")
        for d in docs:
            print(f"  - {d.title}: {d.content[:50]}...")

        print("\n=== 输出状态 ===")
        notes = new_state.get("notes", [])
        print(f"Notes: {len(notes)} 条")
        for n in notes:
            print(f"  - {n.claim_id}: {n.quote} (relevance={n.relevance}, stance={n.stance})")
        messages = new_state.get("messages", [])
        if messages:
            print(f"Message: {messages[0].content}")

    asyncio.run(_demo())
