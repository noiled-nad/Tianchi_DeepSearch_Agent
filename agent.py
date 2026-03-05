# -*- coding: utf-8 -*-
"""
DeepResearch Agent - AgentScope 平台入口

遵循 MIGRATION_GUIDE.md 的最佳实践：
1. 延迟导入策略 - 在 init/query 函数内部导入复杂组件
2. 简化 @agent_app.init - 只做基础初始化
3. 避免使用 framework="langgraph" 参数
4. 模块顶层只导入最基本的组件
"""
from typing import AsyncIterator, List

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from langchain_core.messages import BaseMessage

# 全局状态管理 - 不使用类型注解避免触发导入
short_term_memory = None
long_term_memory = None

# 创建应用实例
agent_app = AgentApp(
    app_name="DeepResearchAgent",
    app_description="A LangGraph-based deep research assistant with iterative retrieval",
)


@agent_app.init
async def initialize(self):
    """
    只做基础初始化，不进行复杂组件的初始化。
    遵循 MIGRATION_GUIDE.md 的建议，避免在初始化阶段编译图结构。
    """
    global short_term_memory, long_term_memory

    # 在 init 内部导入 LangGraph 基础组件
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    short_term_memory = MemorySaver()
    long_term_memory = InMemoryStore()


@agent_app.query
async def query_func(
    self,
    msgs: List[BaseMessage],
    request: AgentRequest = None,
    **kwargs,
) -> AsyncIterator[tuple[BaseMessage, bool]]:
    """
    延迟导入和初始化复杂组件。
    在实际请求处理时才构建 LLM、搜索器、抓取器和图结构。
    使用缓存避免重复创建。
    """
    # 延迟导入 LangGraph 相关组件
    from deepresearch.config import create_llm
    from deepresearch.graph import build_deepresearch_graph
    from deepresearch.tools.search_tool import build_searcher
    from deepresearch.tools.fetch_tool import build_fetcher

    session_id = request.session_id
    user_id = request.user_id
    print(f"Received query from user {user_id} with session {session_id}")

    # 延迟初始化：首次请求时构建图，后续请求复用
    if not hasattr(self, "graph") or self.graph is None:
        llm = create_llm()
        searcher = build_searcher()
        fetcher = build_fetcher()

        graph_builder = build_deepresearch_graph(llm, searcher, fetcher)
        self.graph = graph_builder.compile(checkpointer=short_term_memory, store=long_term_memory)

    # LangGraph thread_id 用于短期记忆/检查点
    config = {"configurable": {"thread_id": session_id}}

    # 执行图
    result_state = await self.graph.ainvoke(
        {"messages": msgs, "session_id": session_id, "user_id": user_id},
        config=config,
    )

    # 最后一条消息就是 finalize 写入的最终回答
    final_msg = result_state["messages"][-1]
    yield final_msg, True


@agent_app.endpoint("/short-term-memory/{session_id}", methods=["GET"])
async def get_short_term_memory(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    value = await short_term_memory.aget_tuple(config)
    if value is None:
        return {"error": "No memory found for session_id"}

    return {
        "session_id": session_id,
        "messages": value.checkpoint["channel_values"]["messages"],
        "metadata": value.metadata,
    }


@agent_app.endpoint("/long-term-memory/{user_id}", methods=["GET"])
async def get_long_term_memory(user_id: str):
    namespace = (user_id, "memories")
    items = long_term_memory.search(namespace)

    def serialize(item):
        return {
            "namespace": item.namespace,
            "key": item.key,
            "value": item.value,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "score": item.score,
        }

    return [serialize(i) for i in items]


if __name__ == "__main__":
    agent_app.run()
