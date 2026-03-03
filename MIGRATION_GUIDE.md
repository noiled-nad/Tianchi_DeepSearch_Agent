# 复杂项目迁移到 AgentScope 平台的经验总结

## 核心问题分析

### 1. Lifespan 参数冲突
- **问题**：平台内部已经处理了 FastAPI 的 `lifespan` 参数，而项目中的 LangGraph 组件或其他库可能也尝试处理生命周期管理，导致 "got multiple values for keyword argument 'lifespan'" 错误。
- **原因**：AgentScope 平台在创建 ASGI 应用时已经内置了生命周期管理，如果代码中再次引入可能涉及生命周期的组件，就会产生冲突。

### 2. 框架初始化时机不当
- **问题**：在应用初始化阶段就进行复杂组件（如 LangGraph 图编译）的初始化，可能导致平台与组件之间的生命周期管理冲突。
- **表现**：平台无法正确加载应用，连接失败。

## 解决方案

### 1. 延迟导入策略
```python
# 在函数内部导入，而不是在模块顶层导入
@agent_app.query
async def query_func(self, msgs, request, **kwargs):
    # 延迟导入 LangGraph 相关组件
    from deepresearch.config import create_llm
    from deepresearch.graph import build_deepresearch_graph
    # ...
```

### 2. 延迟初始化策略
- 不要在 `@agent_app.init` 中进行复杂组件的初始化
- 将组件（如 LangGraph 图）的编译和初始化推迟到实际请求处理时

### 3. 简化初始化过程
- `@agent_app.init` 函数只做最基础的初始化工作
- 避免在此阶段引入任何可能涉及生命周期管理的复杂组件

### 4. 全局状态管理
- 将需要持久化的组件（如 MemorySaver）定义为全局变量
- 在 init 函数中初始化这些全局变量，但不涉及复杂组件

## 迁移最佳实践

### 1. 项目结构适配
- 保持原有项目结构不变
- 创建专门的 `agent.py` 文件作为平台入口
- 通过延迟导入的方式整合原项目功能

### 2. 逐步调试策略
- 先创建最简单的 AgentApp 结构测试平台兼容性
- 逐步添加功能直到找到问题所在
- 使用最小可行代码验证每个组件

### 3. 平台兼容性要点
- 避免使用 `framework="langgraph"` 等可能引起框架冲突的参数
- 不要在初始化阶段预编译复杂图结构
- 遵循平台的生命周期管理机制，不要试图覆盖

### 4. 网络访问配置
- 搜索服务需要通过代理访问网络资源
- HTTP 代理端口：1080
- SOCKS 代理端口：1081
- 配置网络请求使用适当的代理设置以确保外部API调用正常工作

### 5. 错误诊断技巧
- 注意错误信息中的关键词（如 lifespan）
- 理解错误发生在平台加载阶段还是应用执行阶段
- 识别是平台问题还是代码问题

## 可复用的模板结构

```python
# -*- coding: utf-8 -*-
import os
from typing import AsyncIterator, List

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from langchain_core.messages import BaseMessage

# 全局状态管理
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

short_term_memory = None
long_term_memory = None

# 创建应用实例
agent_app = AgentApp(
    app_name="YourAgent",
    app_description="Description",
)

@agent_app.init
async def initialize(self):
    """只做基础初始化"""
    global short_term_memory, long_term_memory
    short_term_memory = MemorySaver()
    long_term_memory = InMemoryStore()

@agent_app.query
async def query_func(self, msgs, request, **kwargs):
    """延迟导入和初始化复杂组件"""
    # 在这里导入原项目的组件
    from your_project.config import create_llm
    from your_project.graph import build_graph
    
    # 执行实际业务逻辑
    # ...
```

这套方法可以适用于大多数将复杂 LangGraph 或其他框架项目迁移到 AgentScope 平台的场景，关键是理解并尊重平台的生命周期管理机制。