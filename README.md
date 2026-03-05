# Tianchi DeepResearch Agent

基于 LangGraph 的 **OAgents 风格子任务拆解 + 级联推理** Deep Research 代理实现：

```
START -> parse_claims(subtask planning) -> execute_subtasks -> finalize -> (execute_subtasks | END)
```

核心思路：将复杂多跳问题拆解为带依赖关系的子任务，按拓扑序分层并行执行，后序子任务注入前序 findings 实现级联推理。

---

## 1. 当前能力

- **子任务拆解**：自动将复杂问题拆分为 2~5 个子任务，标注依赖关系（`depends_on`）
- **分层并行执行**：按拓扑序计算 `parallel_groups`，同层 `asyncio.gather` 并行，跨层级联
- **两阶段查询优化**：Batch Reflection（识别 4 类问题并重写/拆分）+ Parallel Rollout（多样化变体扩展）
- **双模型架构**：主模型（`qwen3.5-plus`）用于规划和综合，轻量模型（`qwen-flash`）用于查询优化和信息抽取
- **增量 followup**：finalize 判断证据不足时，保留已完成子任务 findings，仅新增 gap 子任务定向补查
- **多源搜索聚合**：支持 Serper / 阿里云 IQS / DuckDuckGo / Wikipedia / Bocha
- **抓取后端可切换**：SimpleFetcher / JinaReaderFetcher / HybridFetcher

### 🆕 新增特性（OAgents 风格改进）

| 特性 | 说明 |
|------|------|
| **SubtaskResult 结构化** | 每个子任务记录完整执行元数据（`duration_ms`、`llm_calls`、`success`、`error`） |
| **反思机制（Reflection）** | 每轮迭代自动评估执行质量（`progress_score`），生成改进建议注入下一轮 |
| **Memory 体系** | 完整记录执行轨迹（`ResearchStep` + `ToolCall`），支持 `to_messages()` 转换和上下文压缩 |

---

## 2. 流程架构

主图定义在 [deepresearch/graph.py](deepresearch/graph.py)，3 节点流程：

```
START -> parse_claims -> execute_subtasks -> finalize
                                               |
                              (needs_followup) ↓ (done)
                           execute_subtasks    END
```

### 2.1 parse_claims — 子任务规划

文件：[deepresearch/nodes/parse_claims.py](deepresearch/nodes/parse_claims.py)

- 输入问题，生成 `research_brief`（目标、答案格式、关键实体、约束条件）
- 将问题拆解为多个 `subtasks`，每个子任务包含：
  - `id`：唯一标识（如 ST1, ST2）
  - `title`：子任务标题
  - `reason`：为什么需要这个子任务
  - `queries`：2~4 条初始搜索查询（短查询，每条聚焦一个实体/概念）
  - `depends_on`：前置依赖子任务 ID 列表
- 通过 `_compute_parallel_groups()` 拓扑排序，计算可并行执行的分组
- 注入 `plan_tips`（基于问题关键词匹配的经验规则）
- 创建 `ResearchStep`（plan 类型）记录到 `ResearchMemory`

**查询编写规则**（防止混合检索）：
- 每条查询只检索一个实体/概念
- 依赖子任务的 queries 写成"单侧检索"形式
- 覆盖中英双语，3~8 词短查询

### 2.2 execute_subtasks — 子任务执行

文件：[deepresearch/nodes/execute_subtasks.py](deepresearch/nodes/execute_subtasks.py)

按 `parallel_groups` 分层执行，每个子任务内部走完整流程：

```
query_optimize → search → fetch → extract_findings
```

**子任务内部流程详解：**

1. **依赖注入**：收集前序子任务的 `findings`，构建 `deps_context`
2. **查询细化**（有依赖时）：通过 `DEPS_QUERY_REFINE_PROMPT` 用前序 findings 中的具体实体替换泛化描述
3. **Batch Reflection**：一次 LLM 调用评估所有查询，识别 4 类问题：
   - 信息歧义 → 扩展或去除歧义
   - 语义歧义 → 基于上下文消歧
   - 复杂需求 → **拆分为多条单目标查询**（`|` 分隔符）
   - 过度具体 → 放宽范围
4. **Parallel Rollout**：每条 reflected 查询并行扩展 2~3 个变体（中英双语、同义词、锚点词）
5. **搜索 + 抓取**：多源搜索、URL 去重、并发抓取，记录 `ToolCall` 到 `ResearchStep`
6. **Findings 抽取**：用 flash_llm 从文档中提取与子任务相关的关键事实（3~8 条要点）
7. **返回 SubtaskResult**：包含 `findings`、`duration_ms`、`llm_calls`、`success` 等元数据

**关键设计：**
- 传给 reflect/rollout 的上下文是**子任务 title+reason**（非完整原始问题），防止查询跑偏到其他子任务
- 保留原始查询参与检索，防止优化阶段的语义漂移
- 已有 findings 的子任务自动跳过（支持增量 followup）
- 每个子任务最多 10 条优化查询，最多 6 篇文档

### 2.3 finalize — 综合推理

文件：[deepresearch/nodes/finalize.py](deepresearch/nodes/finalize.py)

- 汇总所有子任务 findings + 文档引用索引，生成最终答案
- 输出 JSON：`reasoning / final_answer / confidence / needs_followup / research_gaps / followup_queries`
- `reasoning` 串联各子任务推理链路（尤其是多跳依赖）

**🆕 反思机制（Reflection）**：
- 统计成功/失败/无信息子任务比例
- 计算 `progress_score`（0.0~1.0）评估本轮进展
- 生成 `suggestions`（改进建议）注入下一轮 gap 子任务
- 创建 `reflect` 类型的 `ResearchStep`

**Prompt 优化**：
- 有 `subtask_findings` 时，原始文档只附轻量索引（标题+URL），大幅压缩 prompt 体积
- 无 findings 时回退到完整证据包
- 注入前序反思建议和 Memory 摘要
- `max_tokens=2048` 限制输出长度

**增量 followup 策略**（非清空重来）：
- 保留原始子任务结构和已完成的 findings
- 为每个 `research_gap` 创建独立 gap 子任务（`gap_0_0`, `gap_0_1`...），带定向查询
- gap 子任务的 `reason` 会注入反思建议
- 重算 `parallel_groups`，已完成子任务由 execute_subtasks 自动跳过
- 无 gaps 但有 followup_queries 时，创建通用 `followup_N` 子任务

### 2.4 query_optimize — 查询优化引擎

文件：[deepresearch/nodes/query_optimize.py](deepresearch/nodes/query_optimize.py)

被 execute_subtasks 内部调用的两阶段查询优化流水线（参考 OAgents SearchReflector）：

| 阶段 | 函数 | 说明 |
|---|---|---|
| Batch Reflection | `_reflect_batch()` | 一次 LLM 调用评估所有查询，识别问题并重写；支持 `\|` 分隔输出拆分复合查询 |
| Parallel Rollout | `_rollout_query()` | 每条查询并行扩展 N 个变体，覆盖同义词/双语/锚点词 |
| Result Reflection | `reflect_search_results()` | 对搜索结果进行相关性评分（similarity + idx 加权） |

**防语义漂移机制**：
- Prompt 明确要求保留领域关键术语，不做类别替换
- 不确定同义时保留原词
- 原始查询始终参与最终查询集

---

## 3. 🆕 Memory 体系

文件：[deepresearch/memory.py](deepresearch/memory.py)

### 3.1 核心数据结构

```python
@dataclass
class ToolCall:
    """工具调用记录"""
    name: str           # 工具名称（search / fetch）
    arguments: Dict     # 调用参数
    result: str         # 返回结果摘要
    error: str          # 错误信息（如有）
    duration_ms: float  # 耗时

@dataclass
class ResearchStep:
    """研究步骤记录"""
    step_type: Literal["plan", "subtask", "search", "fetch", "extract", "finalize", "reflect"]
    step_id: str
    input_summary: str
    output_summary: str
    tool_calls: List[ToolCall]
    duration_ms: float
    success: bool
    score: Optional[float]     # 反思评分
    reflection: Optional[str]  # 反思内容
    sub_steps: List[str]       # 子步骤引用

class ResearchMemory:
    """研究记忆管理器"""
    steps: List[ResearchStep]

    def to_messages(summary_mode, max_steps)  # 转换为 prompt 可用格式
    def compress(max_steps)                   # 压缩长上下文
    def get_statistics()                      # 获取执行统计
    def replay(detailed)                      # 调试回放
```

### 3.2 使用示例

```python
# 输出示例
[execute_subtasks] memory stats: 35 steps, success_rate=100%, total_tool_calls=203

# 每个子任务输出
[ST1] - 物理学者设计了... (duration=40354ms, llm_calls=8)
[ST2] 提取失败。... (duration=64537ms, llm_calls=14)
```

---

## 4. 搜索层

实现文件：[deepresearch/tools/search_tool.py](deepresearch/tools/search_tool.py)

### 支持的 source spec

- `serper`（或 `google`）
- `iqs`（或 `aliyun_iqs` / `aliyun-iqs`）
- `duckduckgo`（或 `ddg`）
- `wikipedia`（或 `wiki`）
- `bocha`
- 兼容：`serpapi[:engine]`、`bing`、`baidu`、`yahoo`（需 SerpApi key）

### 多源聚合行为

- 并发请求各源
- 统一转成 `SearchResult`
- URL 归一化去重，交错合并（round-robin）提升来源多样性
- 运行时输出 `source_raw_counts` 与 `source_errors`

---

## 5. 抓取层

实现文件：[deepresearch/tools/fetch_tool.py](deepresearch/tools/fetch_tool.py)

### 可用抓取器

| 抓取器 | 说明 |
|---|---|
| `SimpleFetcher` | HTML（BeautifulSoup）+ PDF（pypdf）抽取，支持按 query 做相关段提取 |
| `JinaReaderFetcher` | 通过 Jina Reader 网关抓正文，支持 engine/return_format/token_budget |
| `HybridFetcher` | 顺序回退：Simple → Jina(direct) → Jina(browser) |

`build_fetcher()` 根据环境变量 `FETCH_MODE` 选择模式。

---

## 6. 环境变量配置

请在项目根目录创建 `.env`。

### 6.1 LLM

```env
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_KEY
DEEPRESEARCH_MODEL=qwen3.5-plus
DEEPRESEARCH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPRESEARCH_TEMPERATURE=0.2
ENABLE_THINKING=1
MAX_TOKENS=2000

# Flash 模型（查询优化 / findings 抽取用）
FLASH_MODEL=qwen-flash
FLASH_TEMPERATURE=0.3
```

### 6.2 多源搜索

```env
SEARCH_SOURCES=serper,iqs,duckduckgo,wikipedia
SEARCH_PER_SOURCE_RESULTS=5
SEARCH_MAX_RESULTS=8
SEARCH_TIMEOUT_S=20

# Serper
SERPER_API_KEY=YOUR_SERPER_KEY
SERPER_SAFE=active   # active / off

# IQS
IQS_API_KEY=YOUR_IQS_KEY
IQS_BASE_URL=https://cloud-iqs.aliyuncs.com
IQS_ENGINE_TYPE=Generic
IQS_AUTH_MODE=bearer # bearer / x-api-key

# Optional
BOCHA_API_KEY=
SERPAPI_API_KEY=
```

### 6.3 抓取

```env
FETCH_TIMEOUT_S=20
FETCH_MAX_CHARS=12000

# simple | jina | hybrid
FETCH_MODE=simple

JINA_READER_BASE_URL=https://r.jina.ai/http://
JINA_API_KEY=
JINA_ENGINE=direct
JINA_RETURN_FORMAT=text
JINA_TOKEN_BUDGET=50000

# query 相关段提取参数
FETCH_QUERY_TOPK=3
FETCH_QUERY_CHUNK_SIZE=800
FETCH_QUERY_CHUNK_OVERLAP=120
```

---

## 7. 运行方式

### 7.1 单题评测

```bash
python run_one_eval.py
```

输出包括：子任务规划 → 各子任务 findings（含 duration_ms、llm_calls）→ 文档列表 → 最终答案。

### 7.2 启动 AgentApp（可选）

入口在 [app.py](app.py)，使用 agentscope runtime 部署。

---

## 8. 关键文件索引

| 文件 | 用途 |
|---|---|
| [deepresearch/config.py](deepresearch/config.py) | 配置管理，`create_llm()` + `create_flash_llm()` |
| [deepresearch/state.py](deepresearch/state.py) | LangGraph 状态定义（含 subtasks/parallel_groups/subtask_findings/reflections/memory） |
| [deepresearch/graph.py](deepresearch/graph.py) | 3 节点图 + 条件路由 |
| [deepresearch/memory.py](deepresearch/memory.py) | 🆕 Memory 体系（ResearchStep + ToolCall + ResearchMemory） |
| [deepresearch/schemas.py](deepresearch/schemas.py) | 🆕 SubtaskResult 结构化结果定义 |
| [deepresearch/nodes/parse_claims.py](deepresearch/nodes/parse_claims.py) | 子任务规划 + 拓扑排序 + Memory 集成 |
| [deepresearch/nodes/execute_subtasks.py](deepresearch/nodes/execute_subtasks.py) | 子任务执行器（核心节点）+ SubtaskResult + Memory |
| [deepresearch/nodes/query_optimize.py](deepresearch/nodes/query_optimize.py) | 查询优化引擎（reflect + rollout） |
| [deepresearch/nodes/finalize.py](deepresearch/nodes/finalize.py) | 综合推理 + 增量 followup + 反思机制 |
| [deepresearch/nodes/retrieve.py](deepresearch/nodes/retrieve.py) | 搜索+抓取基础逻辑（被 execute_subtasks 内部复用） |
| [deepresearch/plan_tips.py](deepresearch/plan_tips.py) | 关键词匹配的规划经验规则 |
| [deepresearch/tools/search_tool.py](deepresearch/tools/search_tool.py) | 多源搜索聚合 |
| [deepresearch/tools/fetch_tool.py](deepresearch/tools/fetch_tool.py) | 抓取器（Simple/Jina/Hybrid） |

---

## 9. 常见问题

### Q1: 为什么某搜索源显示 0 条？

看 `source_errors`。例如 `wikipedia: ConnectTimeout` 是网络可达性问题。

### Q2: finalize 报 `data_inspection_failed` (400)

输入证据可能触发了风控。缓解方式：
- 开启 `SERPER_SAFE=active`
- 缩短 `FETCH_MAX_CHARS`

### Q3: 子任务的查询混合了多个检索目标怎么办？

系统已内置三层防护：
1. `parse_claims` 的查询编写规则要求每条查询只检索一个实体
2. `DEPS_QUERY_REFINE_PROMPT` 明确禁止混合检索
3. `BATCH_REFLECTION_PROMPT` 对 Complex Requirements 强制拆分（`|` 分隔输出）

### Q4: followup 会重跑所有子任务吗？

不会。已完成子任务的 findings 保留在 state 中，execute_subtasks 自动跳过已有 findings 的子任务，只执行新增的 gap 子任务。

### Q5: 如何查看执行轨迹？

Memory 体系提供多种方式：
- 运行时输出：`memory stats: 35 steps, success_rate=100%, total_tool_calls=203`
- 调试回放：`memory.replay(detailed=True)`
- 转换为消息：`memory.to_messages(summary_mode=True, max_steps=5)`

---

## 10. 改进历史

### v1.1 - OAgents 风格改进

| 改进点 | 说明 |
|--------|------|
| SubtaskResult | 结构化子任务执行结果，记录 `duration_ms`、`llm_calls` |
| 反思机制 | 每轮评估 `progress_score`，生成改进建议 |
| Memory 体系 | 完整记录执行轨迹，支持 `to_messages()` 和 `compress()` |
| Flash 模型 | 从 `qwen3.5-flash` 升级到 `qwen-flash` |
