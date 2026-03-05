# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Tianchi DeepSearch Agent - 基于 LangGraph 的迭代式检索-综合研究代理，采用 **OAgents 风格子任务拆解架构**，用于深度研究和问答任务。

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

# 测试查询优化
python deepresearch/nodes/query_optimize.py
```

## 核心架构

### LangGraph 流程（OAgents 风格子任务版）

```
START -> parse_claims -> execute_subtasks -> finalize -> (execute_subtasks | END)
```

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **parse_claims** | 子任务规划器 | `question` | `subtasks[]`, `parallel_groups`, `research_brief` |
| **execute_subtasks** | 子任务执行器 | `subtasks[]`, `parallel_groups` | `documents[]`, `subtask_findings{}` |
| **finalize** | 答案汇总器 | `subtask_findings{}`, `documents[]` | `final_answer`, `needs_followup`, `research_gaps` |

### 执行流程详解

1. **parse_claims（子任务规划）**
   - 提炼研究简报（objective, key_entities, hard_constraints）
   - 将问题拆解为多个子任务（subtasks），标注依赖关系（depends_on）
   - 计算并行执行组（parallel_groups）- 同组可并行，跨组有依赖

2. **execute_subtasks（子任务执行）**
   - 按 parallel_groups 分层执行
   - 每个子任务内部流程：`query_optimize → search → fetch → extract_findings`
   - 后序子任务可注入前序 findings（级联推理）
   - 组内 asyncio.gather 并行

3. **finalize（答案汇总）**
   - 汇总所有子任务的 findings
   - 判断是否需要继续检索（needs_followup）
   - 如需继续，产出 follow-up queries 和 gap 子任务

迭代由 `needs_followup` 和 `iteration < max_iterations` 控制，默认最多 4 轮。

### 状态定义 (deepresearch/state.py)

```python
class DeepResearchState(TypedDict, total=False):
    # 对话上下文
    messages: Annotated[List[BaseMessage], add_messages]

    # 问题
    question: str
    research_brief: Dict[str, Any]  # objective, key_entities, hard_constraints

    # 子任务拆解
    subtasks: List[Dict[str, Any]]           # [{id, title, queries, depends_on, reason}]
    parallel_groups: List[List[str]]         # [["ST1","ST2"], ["ST3"]]
    subtask_findings: Dict[str, str]         # {"ST1": "findings...", "ST2": "..."}

    # 搜索内容
    queries: List[str]
    query_history: List[str]
    research_gaps: List[str]
    needs_followup: bool

    # 结果
    documents: List[Document]
    final_answer: str

    # 调度参数
    iteration: int
    max_iterations: int
```

### 关键模块

| 模块 | 职责 |
|------|------|
| `deepresearch/config.py` | 配置加载、LLM 创建（支持 enable_thinking、flash_llm） |
| `deepresearch/graph.py` | StateGraph 定义和条件路由 |
| `deepresearch/plan_tips.py` | OAgents 风格启发式规则库（Plan Tips） |
| `deepresearch/nodes/parse_claims.py` | 子任务规划节点 |
| `deepresearch/nodes/execute_subtasks.py` | 子任务执行节点（并行+级联） |
| `deepresearch/nodes/finalize.py` | 答案汇总节点 |
| `deepresearch/nodes/query_optimize.py` | 查询优化（Reflection + Rollout） |
| `deepresearch/tools/search_tool.py` | 多源搜索聚合（Serper/IQS/DuckDuckGo/Wikipedia/Bocha） |
| `deepresearch/tools/fetch_tool.py` | 抓取器（Simple/Jina/Hybrid） |

## 环境配置

必需的环境变量（通过 `.env` 配置）：

```env
# LLM 配置
DASHSCOPE_API_KEY=your_key           # 必需
DEEPRESEARCH_MODEL=qwen3.5-plus
FLASH_MODEL=qwen3.5-flash            # 用于 query_optimize 的轻量推理

# 搜索配置
SEARCH_SOURCES=serper,iqs,duckduckgo,wikipedia
SERPER_API_KEY=your_serper_key
IQS_API_KEY=your_iqs_key

# 抓取配置
FETCH_MODE=simple                    # simple | jina | hybrid
```

## 关键设计

### 1. Plan Tips（启发式规则库）

来自 OAgents 论文的核心发现：Plan Tips 贡献最大（+14.54%）。

位置：`deepresearch/plan_tips.py`

- 根据问题特征（年份、金额、多跳、学术等）匹配相关 tips
- 在 parse_claims 时注入 prompt，指导查询生成

### 2. Query Optimization（两阶段流水线）

位置：`deepresearch/nodes/query_optimize.py`

1. **Batch Reflection** - 识别 4 类问题并重写
   - 信息歧义、语义歧义、复杂需求、过度具体
2. **Query Rollout** - 扩展多样化变体
   - 同义词、中英双语、锚点词

### 3. 级联推理（Cascade Reasoning）

- 后序子任务可访问前序子任务的 findings
- 多跳问题的核心价值：先锁定桥接实体，再用桥接实体搜索目标

### 4. 双模型策略

- **主模型**（qwen3.5-plus）：parse_claims, finalize
- **Flash 模型**（qwen3.5-flash）：query_optimize, extract_findings

## AgentScope 平台集成

入口文件：`app.py` 和 `agent.py`

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
- **子任务未执行**：检查 `parallel_groups` 是否正确，已完成的子任务会被跳过

## 参考框架

本项目学习自 OAgents（/mnt/workspace/OAgents），主要借鉴：
- 子任务拆解 + 并行执行架构
- Plan Tips 启发式规则库
- Query Reflection + Rollout 优化
- Memory Step 结构化设计（待实现）
