# run_one.py
# -*- coding: utf-8 -*-
import asyncio
import json

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepresearch.config import create_llm, create_flash_llm
from deepresearch.graph import build_deepresearch_graph
from deepresearch.tools.search_tool import build_searcher
from deepresearch.tools.fetch_tool import build_fetcher


TEST = {"id": 1, 
        "question": "一位物理学领域的学者为一种经典棋盘游戏设计的评分系统，后来被一家北美游戏公司广泛应用于其一款多人在线战术竞技游戏中。这家公司的母公司是一家亚洲科技巨头，该巨头在21世纪10年代完成了对前者的全资收购，并涉足量子计算等前沿科技领域。在这家北美公司开发的另一款第一人称射击游戏中，有一件适合近距离作战的武器，其名称与上述亚洲巨头代理发行的一款格斗手游中的一名在登场角色中年龄偏大的武术教官角色相同。这款格斗手游的名字是什么？"
        }

async def main():
    # 1) 初始化：LLM + 工具 + 图
    llm = create_llm()
    flash_llm = create_flash_llm()
    searcher = build_searcher()
    fetcher = build_fetcher()

    graph_builder = build_deepresearch_graph(llm, searcher, fetcher, flash_llm=flash_llm)

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
    print("\n====== [parse_claims 输出] subtasks ======")
    for st in result_state.get("subtasks", []):
        deps = st.get('depends_on', [])
        print(f"  [{st.get('id')}] {st.get('title')}  (deps={deps})")
        for q in st.get("queries", []):
            print(f"       -> {q}")

    print("\n====== [subtask_findings 输出] ======")
    for st_id, findings in (result_state.get("subtask_findings", {}) or {}).items():
        print(f"  [{st_id}] {findings[:200]}")

    print("\n====== [retrieve 输出] documents URLs ======")
    docs = result_state.get("documents", [])
    for i, d in enumerate(docs, start=1):
        title = (d.title or "").strip().replace("\n", " ")
        print(f"[S{i}] {title} | {d.url}")

    print("\n====== [finalize 输出] final_answer ======")
    print(result_state.get("final_answer", ""))




if __name__ == "__main__":
    asyncio.run(main())
