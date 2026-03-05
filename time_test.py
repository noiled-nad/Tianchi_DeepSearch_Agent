# -*- coding: utf-8 -*-
"""Quick benchmark for qwen3.5-plus request latency.

Usage:
  python bench_qwen35plus_speed.py
  python bench_qwen35plus_speed.py --rounds 5 --mode both
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


SHORT_PROMPT = "请用一句话解释什么是Elo评分系统。"

LONG_PROMPT = (
    "你是任务规划器。请将下面问题拆成2-4个子任务，返回JSON。\n"
    "问题：一位物理学领域的学者为一种经典棋盘游戏设计的评分系统，后来被一家北美游戏公司"
    "广泛应用于其一款多人在线战术竞技游戏中。这家公司母公司是一家亚洲科技巨头。"
)


def build_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("DEEPRESEARCH_MODEL", "qwen3.5-plus"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DEEPRESEARCH_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=float(os.getenv("DEEPRESEARCH_TEMPERATURE", "0.2")),
    )


async def one_non_stream(llm: ChatOpenAI, prompt: str) -> Tuple[float, int]:
    t0 = time.perf_counter()
    resp = await llm.ainvoke(prompt)
    total = time.perf_counter() - t0
    out = str(getattr(resp, "content", "") or "")
    return total, len(out)


async def one_stream(llm: ChatOpenAI, prompt: str) -> Tuple[float, float, int, int]:
    t0 = time.perf_counter()
    first_token = None
    chunks = 0
    out_len = 0

    async for chunk in llm.astream(prompt):
        txt = str(getattr(chunk, "content", "") or "")
        if not txt:
            continue
        if first_token is None:
            first_token = time.perf_counter() - t0
        chunks += 1
        out_len += len(txt)

    total = time.perf_counter() - t0
    return (first_token if first_token is not None else total), total, chunks, out_len


async def run_case(llm: ChatOpenAI, name: str, prompt: str, rounds: int) -> None:
    print(f"\n=== Case: {name} | prompt_len={len(prompt)} | rounds={rounds} ===")

    ns_times: List[float] = []
    for i in range(rounds):
        t, out_len = await one_non_stream(llm, prompt)
        ns_times.append(t)
        print(f"[non-stream #{i+1}] total={t:.2f}s, out_len={out_len}")

    st_first: List[float] = []
    st_total: List[float] = []
    for i in range(rounds):
        first, total, chunks, out_len = await one_stream(llm, prompt)
        st_first.append(first)
        st_total.append(total)
        print(
            f"[stream #{i+1}] first_token={first:.2f}s, total={total:.2f}s, "
            f"chunks={chunks}, out_len={out_len}"
        )

    print("--- summary ---")
    print(
        f"non-stream: avg={statistics.mean(ns_times):.2f}s, "
        f"p50={statistics.median(ns_times):.2f}s, max={max(ns_times):.2f}s"
    )
    print(
        f"stream first-token: avg={statistics.mean(st_first):.2f}s, "
        f"p50={statistics.median(st_first):.2f}s, max={max(st_first):.2f}s"
    )
    print(
        f"stream total: avg={statistics.mean(st_total):.2f}s, "
        f"p50={statistics.median(st_total):.2f}s, max={max(st_total):.2f}s"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mode", choices=["short", "long", "both"], default="both")
    args = parser.parse_args()

    llm = build_llm()

    if args.mode in ("short", "both"):
        await run_case(llm, "short", SHORT_PROMPT, args.rounds)
    if args.mode in ("long", "both"):
        await run_case(llm, "long", LONG_PROMPT, args.rounds)


if __name__ == "__main__":
    asyncio.run(main())
