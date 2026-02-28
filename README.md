# Tianchi DeepResearch Agent

当前仓库是一个可运行的 DeepResearch 代理实现，基于 LangGraph 的**迭代式检索-综合**流程：

`parse_claims(brief) -> retrieve -> finalize -> (retrieve | END)`

与旧版相比，当前主流程不再依赖 SPOQ 强结构化拆分，而是采用 `research_brief + follow-up queries` 的轻量循环研究策略。

---

## 1. 当前能力（按代码实况）

- 迭代研究图：最多多轮检索，`finalize` 决定是否继续。
- 多源搜索聚合：支持 `Serper / 阿里云 IQS / DuckDuckGo / Wikipedia / Bocha`，并兼容 `SerpApi`。
- 搜索并发聚合去重：跨源并发、URL 去重、交错合并。
- 搜索可观测性：输出每个源的 `source_raw_counts` 与 `source_errors`。
- 抓取后端可切换：`SimpleFetcher / JinaReaderFetcher / HybridFetcher`。
- Query 相关段抽取：抓取文本可按 query 做轻量打分抽段，降低噪声。

---

## 2. 流程架构

主图定义在 [deepresearch/graph.py](deepresearch/graph.py)：

- `START -> parse_claims -> retrieve -> finalize`
- `finalize` 根据 `needs_followup` 和 `iteration/max_iterations` 路由：
  - 继续检索：`retrieve`
  - 结束：`END`

### 节点说明

- [deepresearch/nodes/parse_claims.py](deepresearch/nodes/parse_claims.py)
  - 输入问题，生成 `research_brief` 和首轮 `queries`（5~8 条）
  - 初始化循环状态：`needs_followup=True`、`iteration/max_iterations`

- [deepresearch/nodes/retrieve.py](deepresearch/nodes/retrieve.py)
  - 对 `queries` 做搜索，收集候选 URL
  - 并发抓取文档，增量合并到 `documents`
  - 支持 `parallelism`、跨轮 URL 去重

- [deepresearch/nodes/finalize.py](deepresearch/nodes/finalize.py)
  - 基于证据包输出 JSON：`final_answer/confidence/needs_followup/research_gaps/followup_queries`
  - 若需继续检索，写回新一轮 `queries`

---

## 3. 搜索层（新增：多源聚合 + Serper 安全搜索）

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
- 结果 snippet 自动打源标签（如 `[serper] ...`）
- URL 归一化去重
- 交错合并（round-robin）提升来源多样性
- 多源时自动保证总上限：`SEARCH_MAX_RESULTS >= SEARCH_PER_SOURCE_RESULTS * 源数量`

### 运行时可观测指标

脚本自测会输出：

- `sources(n)`
- `source_raw_counts`
- `source_errors`（例如 `ConnectTimeout`）

可直接运行：

```bash
python deepresearch/tools/search_tool.py
```

---

## 4. 抓取层（新增：Jina Reader）

实现文件：[deepresearch/tools/fetch_tool.py](deepresearch/tools/fetch_tool.py)

### 可用抓取器

- `SimpleFetcher`
  - HTML 抽取（BeautifulSoup）
  - PDF 抽取（pypdf / PyPDF2）
  - 可按 query 做文本分块与相关段提取

- `JinaReaderFetcher`（新增）
  - 通过 Jina Reader 网关抓正文
  - 支持 `engine`、`return_format`、`token_budget`

- `HybridFetcher`（新增）
  - 顺序回退：`Simple -> Jina(direct) -> Jina(browser)`
  - 提升抓取成功率

`build_fetcher()` 根据环境变量选择模式。

---

## 5. 环境变量配置（建议）

请在项目根目录创建 `.env`。

### 5.1 LLM

```env
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_KEY
DEEPRESEARCH_MODEL=qwen3.5-plus
DEEPRESEARCH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPRESEARCH_TEMPERATURE=0.2
ENABLE_THINKING=1
MAX_TOKENS=2000
```

### 5.2 多源搜索（推荐）

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

### 5.3 抓取（含 Jina Reader）

```env
FETCH_TIMEOUT_S=20
FETCH_MAX_CHARS=12000

# simple | jina | hybrid
FETCH_MODE=simple

# 兼容开关：设置为 jina 或 1 也会启用 JinaReader
FETCH_READER=
USE_JINA_READER=0

JINA_READER_BASE_URL=https://r.jina.ai/http://
JINA_API_KEY=
JINA_ENGINE=direct
JINA_RETURN_FORMAT=text
JINA_TOKEN_BUDGET=50000
JINA_BROWSER_TOKEN_BUDGET=80000

# query 相关段提取参数
FETCH_QUERY_TOPK=3
FETCH_QUERY_CHUNK_SIZE=800
FETCH_QUERY_CHUNK_OVERLAP=120
```

---

## 6. 运行方式

### 6.1 单题评测脚本

```bash
python run_one_eval.py
```

该脚本会构建图并输出最终答案。

### 6.2 启动 AgentApp（可选）

入口在 [app.py](app.py)。如果你使用 agentscope runtime，可按你的部署方式启动该应用。

---

## 7. 关键文件索引

- [deepresearch/config.py](deepresearch/config.py)：加载配置，创建 LLM（支持 `enable_thinking`）
- [deepresearch/state.py](deepresearch/state.py)：LangGraph 状态定义
- [deepresearch/graph.py](deepresearch/graph.py)：图与条件路由
- [deepresearch/nodes/parse_claims.py](deepresearch/nodes/parse_claims.py)：研究简报与首轮 query 生成
- [deepresearch/nodes/retrieve.py](deepresearch/nodes/retrieve.py)：搜索+抓取
- [deepresearch/nodes/finalize.py](deepresearch/nodes/finalize.py)：答案综合与 follow-up 决策
- [deepresearch/tools/search_tool.py](deepresearch/tools/search_tool.py)：多源搜索聚合
- [deepresearch/tools/fetch_tool.py](deepresearch/tools/fetch_tool.py)：抓取器（Simple/Jina/Hybrid）

---

## 8. 常见问题

### Q1: 为什么某源显示 0 条？

看 `source_errors`。例如 `wikipedia: ConnectTimeout` 往往是网络可达性问题，而不是融合逻辑问题。

### Q2: finalize 报 `data_inspection_failed` (400)

通常是输入证据包含触发风控的文本。可通过以下方式缓解：

- 开启 `SERPER_SAFE=active`
- 提升检索源质量（`serper + iqs`）
- 缩短抓取文本上限（`FETCH_MAX_CHARS`）

### Q3: 无 SerpApi 是否可用？

可以。推荐直接使用：`serper,iqs,duckduckgo,wikipedia`。

---

## 9. 本次新增修改说明

1. **Jina Reader 抓取链路**
   - 新增 `JinaReaderFetcher`
   - 新增 `HybridFetcher` 回退策略
   - `build_fetcher()` 支持 `FETCH_MODE=jina/hybrid`

2. **多源搜索增强**
   - 新增 `SerperSearcher`、`AliyunIQSSearcher`
   - `MultiSourceSearcher` 增加每源计数与错误统计
   - 支持 `SEARCH_SOURCES` 灵活组合

3. **安全搜索**
   - `Serper` 支持 `SERPER_SAFE`，默认 `active`
   - `Serper` 仍然存在检索到违禁内容的问题，后续可能需要其他过滤方式

