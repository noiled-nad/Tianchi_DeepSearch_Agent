# run_one.py
# -*- coding: utf-8 -*-
import asyncio
import json

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepresearch.config import create_llm
from deepresearch.graph import build_deepresearch_graph
from deepresearch.tools.search_tool import build_searcher
from deepresearch.tools.fetch_tool import SimpleFetcher


TEST = {"id": 1, 
        "question": "Who is the author of the article that introduces the methodology of prosopography and demographic analysis of colonial social structures, analyzes the structural evolution of encomienda and hacienda systems, critiques historiographical gaps in prior political-centric approaches to colonial Spanish America, and was first published in the 1972 journal issue that analyzes the economic impacts of import substitution industrialization policies in Latin America and critiques political-centric historiographical approaches to colonial development?"
        }

async def main():
    # 1) 初始化：LLM + 工具 + 图
    llm = create_llm()
    searcher = build_searcher()
    fetcher = SimpleFetcher(timeout_s=20.0)

    graph_builder = build_deepresearch_graph(llm, searcher, fetcher)

    # baseline：先用内存 checkpoint/store
    graph = graph_builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())

    # 2) 组装输入
    question = TEST["question"]
    msgs = [HumanMessage(content=question)]

    # thread_id 用于 checkpoint
    config = {"configurable": {"thread_id": f"test-{TEST['id']}"}}

    # 3) 跑图
    result_state = await graph.ainvoke({"messages": msgs}, config=config)

    # 4) 打印每一步产物
    print("\n====== [parse_claims 输出] claims ======")
    for c in result_state.get("claims", []):
        print(f"{c.id}: {c.description}")

    print("\n====== [plan_queries 输出] queries ======")
    for q in result_state.get("queries", []):
        print("-", q)

    print("\n====== [retrieve 输出] documents URLs ======")
    docs = result_state.get("documents", [])
    for i, d in enumerate(docs, start=1):
        title = (d.title or "").strip().replace("\n", " ")
        print(f"[S{i}] {title} | {d.url}")

    print("\n====== [finalize 输出] final_answer ======")
    print(result_state.get("final_answer", ""))




if __name__ == "__main__":
    asyncio.run(main())
