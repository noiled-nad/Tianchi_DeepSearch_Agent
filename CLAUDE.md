# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Tianchi DeepSearch Agent - 基于 LangGraph 的迭代式检索-综合研究代理，用于深度研究和问答任务。

## 常用命令

```bash
# 单题评测（开发调试首选）
python run_one_eval.py

# 启动 AgentScope Runtime 服务
python app.py

# 测试搜索工具
python deepresearch/tools/search_tool.py

# 生成图结构 Mermaid 图
python deepresearch/graph.py
```

## 核心架构

### LangGraph 流程
```
START -> parse_claims -> retrieve -> finalize -> (retrieve | END)
```

- **parse_claims**: 生成 research_brief 和首轮 queries（5-8条）
- **retrieve**: 多源搜索 + 并发抓取，增量合并到 documents
- **finalize**: 输出 JSON（final_answer/confidence/needs_followup），决定是否继续检索

迭代由 `needs_followup` 和 `iteration < max_iterations` 控制，默认最多 4 轮。

### 状态定义 (deepresearch/state.py)
- `DeepResearchState`: 核心状态 TypedDict，包含 messages、question、queries、documents、final_answer 等
- 使用 `Annotated[List[BaseMessage], add_messages]` 管理对话消息

### 关键模块

| 模块 | 职责 |
|------|------|
| `deepresearch/config.py` | 配置加载、LLM 创建（支持 enable_thinking） |
| `deepresearch/graph.py` | StateGraph 定义和条件路由 |
| `deepresearch/tools/search_tool.py` | 多源搜索聚合（Serper/IQS/DuckDuckGo/Wikipedia） |
| `deepresearch/tools/fetch_tool.py` | 抓取器（Simple/Jina/Hybrid） |
| `deepresearch/nodes/` | 图节点实现 |

## 环境配置

必需的环境变量（通过 `.env` 配置）：

```env
DASHSCOPE_API_KEY=your_key           # 必需
DEEPRESEARCH_MODEL=qwen3.5-plus
SEARCH_SOURCES=serper,iqs,duckduckgo,wikipedia
SERPER_API_KEY=your_serper_key
IQS_API_KEY=your_iqs_key
```

## AgentScope 平台集成

入口文件：`app.py`

**重要限制**：
- 平台内部已处理 FastAPI lifespan，避免在 `@agent_app.init` 中进行复杂组件初始化
- 使用延迟导入策略，将 LangGraph 组件的导入推迟到 query 函数内部
- 如遇 "got multiple values for keyword argument 'lifespan'" 错误，检查初始化流程

## 网络代理

搜索服务需要通过代理访问外部资源：
- HTTP 代理：`127.0.0.1:1080`
- SOCKS 代理：`127.0.0.1:1081`

Python 中使用：
```python
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'
```

## 故障排查

- **某搜索源返回 0 条**：检查 `source_errors`，通常是网络超时
- **finalize 报 data_inspection_failed (400)**：开启 `SERPER_SAFE=active`，缩短 `FETCH_MAX_CHARS`
- **图编译错误**：确保节点返回的是状态更新字典，而非完整状态对象
