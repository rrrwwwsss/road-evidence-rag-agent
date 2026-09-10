# 非现场取证线索库智能查询系统

> 基于 LangGraph + Intent-Skill-Tool 分层架构的道路/现场取证智能查询系统，集成 **RAG 知识库（Dify）**、**SQL 案件数据库查询**、**Qwen-VL 图片违法识别** 与 **执法报告生成**。

---

## 目录

- [项目简介](#项目简介)
- [核心功能与架构](#核心功能与架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [配置说明](#配置说明)
- [RAG 检索评估](#rag-检索评估)
- [注意事项与安全](#注意事项与安全)
- [技术栈](#技术栈)

---

## 项目简介

系统面向道路/现场取证、证据检索与执法辅助场景。Intent 层识别一个或多个用户目标与实体，Planner 生成带依赖关系的执行计划，Executor 顺序调用业务 Skill，Skill 再按固定规则调用原子 Tool：

- 🧠 **RAG 知识库检索**：判定标准、法规依据、历史案例相似度（对接 Dify 知识库，支持 `doc_type=standard/case` 元数据过滤）；
- 🗄️ **SQL 案件查询**：自然语言转只读 SQL，查询案件数据库（计数、按时间/支队/地点/违法类型统计）；
- 🖼️ **视觉违法识别**：上传图片调用 Qwen-VL 识别违法行为，可结合 RAG 判定标准进行“是否属于道路养护”等判定；
- 📋 **执法报告生成**：基于 SQL 统计 + RAG 证据自动生成 Markdown 执法/取证报告草稿。

---

## 核心功能与架构

### 整体架构

```
                Streamlit Web UI (app.py，支持文字 + 图片上传)
                               │ 用户输入
                               ▼
              Intent Classifier（意图与实体识别）
                               │
                               ▼
       LangGraph → Planner → ExecutionPlan → Skill 路由
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
   案件/知识 Skill       图片判定 Skill          报告 Skill
          │                    │                     │
          └────────────── Structured Tools ──────────┘
                               ▼
                  生成最终回答 / Markdown 报告
```

### Intent、Skill 与 Tool

`agent/orchestrator.py` 是兼容门面，实际运行时位于 `agent/graph/workflow.py`。LangGraph 将意图识别、计划生成、逐 Skill 路由、依赖跳过和结果汇总编译为独立节点；`agent/planning/` 将多目标请求转换为依赖计划，`agent/executor.py` 提供可由图节点逐步调用的确定性执行能力。`agent/state.py` 保存当前意图、计划、Skill、共享产物、工具结果、证据、错误与事件。业务规则由 Skill 强制执行，不依赖大模型临场选择调用顺序。

短期上下文记忆保留最近 8 轮用户与助手消息。页面展示历史仍由 Streamlit Session State 保存；Agent 使用的有限消息窗口由 `agent/memory.py` 统一裁剪，写入 `AgentState.recent_messages`，并随 LangGraph checkpoint 保存。单条超长消息会截断，系统提示词和无效角色不会进入会话记忆。

系统采用 **Plan-Execute 为主、Skill 内有限纠错为辅** 的混合模式：清晰请求由规则直接规划；复杂请求可由结构化模型补充目标；执行计划必须通过依赖校验。SQL 修正、检索回退等局部循环均设置次数上限，不允许无限 ReAct。复合请求如“先找图片历史案例，再结合案例判断违法”会执行 `ImageCaseSearchSkill → ImageAssessmentSkill`，并复用前一步案例证据。

| 问题类型 | Skill | 固定调用路径 |
| --- | --- | --- |
| 计数/统计/案件明细 | `CaseQuerySkill` | SQL Tool |
| 判定标准/法规依据/历史案例 | `KnowledgeQuerySkill` | RAG Tool |
| 根据图片查找相似案例 | `ImageCaseSearchSkill` | Vision 特征提取 → RAG Tool，不作违法认定 |
| 图片违法或养护判定 | `ImageAssessmentSkill` | RAG Tool → Vision Tool |
| 执法/取证报告 | `ReportGenerationSkill` | SQL Tool → RAG Tool → 报告生成 |

Tool 统一返回 `ToolResult`，包含成功状态、结构化数据、证据来源、错误码、耗时和截断状态。报告流程不再使用状态切换伪工具。

### RAG 知识库（Dify）

- 检索走 Dify Dataset API：`POST /v1/datasets/{dataset_id}/retrieve`，支持 `semantic_search` / `full_text_search` / `hybrid_search`（混合检索需 Dify 配置 rerank 模型）；
- 支持按元数据 `doc_type`（`standard`/`case`）过滤；
- 未配置 Dify（`DIFY_API_KEY` 或 `dataset_id` 为空）时自动回退本地 Chroma 向量库。

### SQL 查询

- `rag/sql_service.py` 内部用大模型把自然语言问题转成只读 SQL（仅放行 `SELECT/WITH`），失败自动修正重试一次；
- 以只读模式（`mode=ro`）连接 SQLite，防止误写数据。

### 视觉识别

- `rag/vision_service.py` 调用 `qwen-vl-max`（DashScope），支持图片 ID / data URI / URL；
- 上传图片由前端生成 `img_xxxx` 标识并保存在当前 Streamlit 会话中；调用 Vision Skill 时通过 LangGraph Runtime Context 显式传入，不依赖跨线程的模块级全局字典，也不会写入 checkpoint。

---

## 项目结构

```
├── app.py                        # Streamlit 入口（文字 + 图片上传）
├── agent/
│   ├── react_agent.py            # 保持旧调用方式的兼容门面
│   ├── orchestrator.py           # Intent → Plan → Execute 编排与异常处理
│   ├── graph/                     # LangGraph 状态图、节点路由与内存 Checkpointer
│   ├── executor.py               # 按依赖确定性执行并汇总 Skill 输出
│   ├── planning/                 # ExecutionPlan / PlanStep 与 Planner
│   ├── state.py                  # 显式执行状态、共享产物、证据和事件模型
│   ├── memory.py                 # 最近 8 轮用户/助手消息窗口
│   ├── intents/                  # 意图分类、实体抽取与数据模型
│   ├── skills/                   # 文件化 Skill 能力包
│   │   ├── case_query/           # SKILL.md + prompt.md + handler.py
│   │   ├── knowledge_query/
│   │   ├── image_case_search/
│   │   ├── image_assessment/
│   │   └── report_generation/
│   ├── policies/                 # 证据完整性与审核结果约束
│   └── tools/
│       ├── contracts.py          # ToolResult / EvidenceSource 契约
│       └── structured_tools.py   # RAG / SQL / Vision 原子 Tool 适配器
├── rag/
│   ├── rag_service.py            # RAG 汇总（Dify 优先，Chroma 兜底）
│   ├── dify_retriever.py         # Dify 知识库检索客户端
│   ├── sql_service.py            # 自然语言转 SQL 查询
│   ├── vision_service.py         # Qwen-VL 视觉分析
│   └── vector_store.py           # 本地 Chroma 向量库构建
├── model/factory.py              # 对话 / 嵌入 / 视觉模型工厂
├── config/
│   ├── agent.yml                 # SQLite 数据库路径
│   ├── rag.yml                   # 模型名配置
│   ├── chroma.yml                # 本地向量库参数
│   ├── dify.yml                  # Dify 知识库配置
│   └── prompts.yml               # 提示词文件路径
├── prompts/
│   ├── main_prompt.txt           # 一般回答与合规提示词
│   └── rag_summarize.txt         # RAG Tool 基础汇总提示词
├── data/
│   ├── records.csv               # 案件原始数据
│   ├── wupin_tanwei_dabt.db      # 案件数据库（SQL 查询数据源）
│   ├── 违法案例库/*.md           # 知识库案例文档
│   ├── 道路养护判定标准.md       # 判定标准（Dify 中 doc_type=standard）
│   └── 公路范围内的合法物品判定标准.md
├── eval/                         # RAG 与 Intent 评估
│   ├── eval_retrieval.py
│   ├── test_set.jsonl
│   ├── eval_intents.py
│   └── intent_test_set.jsonl
├── tests/                        # Skill 路径与编排单元测试
└── 处理代码/                     # 数据处理脚本（csv→md/json、时间戳修正、删表等）
```

---

## 快速开始

### 环境要求

- Python 3.10+（推荐使用 Conda 环境，如 `agent`）
- Dify（自部署或云端）与知识库

### 安装依赖

```bash
pip install -r requirements.txt
# 或按需安装：
pip install streamlit langchain langchain-openai langchain-community langchain-chroma \
            langchain-text-splitters dashscope python-dotenv pyyaml requests pandas
```

### 配置 `.env`

```env
DASHSCOPE_API_KEY=sk-xxx          # 嵌入模型 text-embedding-v4 / 视觉模型 qwen-vl-max
OPENAI_API_KEY=sk-xxx             # 对话模型（DeepSeek 兼容接口）
OPENAI_API_BASE=https://api.deepseek.com
DIFY_API_KEY=dataset-xxx          # Dify 知识库 Dataset API Key
```

### Dify 知识库配置

1. 在 Dify 中创建知识库，上传 `data/违法案例库/*.md`（`doc_type=case`）与 `data/道路养护判定标准.md`、`data/公路范围内的合法物品判定标准.md`（`doc_type=standard`）；
2. 在知识库「API 访问」中生成 Dataset API Key；
3. 编辑 `config/dify.yml`：填入 `api_base`（自部署如 `http://localhost:3272/v1`）与 `dataset_id`（知识库 URL 中的 ID）。

### （可选）本地向量库

未配置 Dify 时使用本地 Chroma 作为兜底，需先构建索引：

```powershell
Remove-Item chroma_db -Recurse -Force
Clear-Content md5.text
python rag/vector_store.py
```

### 启动应用

```bash
streamlit run app.py
```

---

## 使用说明

### Web 界面

- 文字输入：直接提问，如「擅自占用公路一共有多少条？」
- 图片上传：点击聊天输入框的图片按钮上传 jpg/png，然后提问，如「判断一下这是不是道路养护行为」
- 处理过程：回答生成期间会实时展示可审计事件；点击任一步骤可查看事件类别、状态、时间和用途说明。该区域展示计划、Skill、Tool 与证据处理过程，不展示模型内部隐藏思维链。

前端通过 `ReactAgent.astream_events(version="v2")` 消费兼容事件；底层事件来自 LangGraph 原生 `custom/values` 流。事件包含 `on_chain_start`、`on_custom_event` 和 `on_chain_end`；业务进度统一使用 `agent_progress` 自定义事件，并在会话历史中持久化以便再次展开查看。每个 Streamlit 会话绑定稳定的 LangGraph `thread_id`，进程内 Checkpointer 保存节点状态。工具完成后还会生成脱敏的 `tool_observation`：SQL 展示实际只读语句和前 10 行数据，RAG 展示检索参数、摘要和前 5 条来源，Vision 展示图片 ID、分析模式和结果预览。

### 示例问题

- 计数统计：「擅自占用公路一共有多少条？」「2026年7月23日哪个支队最多？」
- 判定标准：「道路养护行为的判定标准是什么？」
- 历史案例：「与擅自占用、挖掘公路类似的历史案例有哪些？」
- 图片识别：上传图片后问「这张图片是什么违法行为？」
- 图片判定：上传图片后问「图中行为是否属于道路养护行为？」
- 复合图片任务：上传图片后问「查找这张图的历史案例，并根据案例和判定标准分析是否存在擅自占用公路行为」
- 报告生成：「请为2026年5月密云执法队的违法案件生成一份执法报告草稿」

---

## 配置说明

| 文件 | 说明 |
| --- | --- |
| `config/rag.yml` | `chat_model_name`（对话）、`embedding_model_name`（嵌入）、`vision_model_name`（视觉，如 `qwen-vl-max`） |
| `config/dify.yml` | Dify `api_base` / `dataset_id` / 检索参数，以及连接超时、读取超时和有限重试配置 |
| `config/agent.yml` | `sqlite_db_path`：SQL 查询的数据源 |
| `config/chroma.yml` | 本地向量库 collection / 分块参数 / `k` |

---

## RAG 检索评估

`eval/` 目录提供**只测检索、不调用大模型**的评估脚本：

```powershell
python eval\eval_retrieval.py                 # 评估 test_set.jsonl，top_k=5
python eval\eval_retrieval.py --top-k 3,5,10  # 对比多个 k
python eval\eval_retrieval.py --query "道路养护行为的判定标准是什么"
python eval\eval_retrieval.py --dump          # 查看每个问题实际召回
```

指标：`HitRate@k / Recall@k / Precision@k / MRR / NDCG@k`，按 `doc_type` 分组统计并输出失败样例。详见 `eval/README.md`。

### Intent 与 Skill 路径评测

```powershell
python eval\eval_intents.py
python -m unittest discover -s tests -v
```

第一条命令基于 `eval/intent_test_set.jsonl` 评估规则分类准确率；第二条验证 Intent、实体抽取、Skill 选择，以及图片和报告场景的强制 Tool 调用顺序。

---

## 注意事项与安全

- **密钥保护**：`.env` 已加入 `.gitignore`，请勿提交；历史版本中若出现过密钥请尽快轮换；
- **执法合规**：系统输出仅作为辅助参考，涉及法律定性的结论需「人工/执法部门复核」；
- **过程可视化边界**：只展示可审计执行事件、工具状态和来源说明，不记录或展示模型隐藏思维链、系统提示词及中间草稿；
- **数据最小化**：报告与对外输出遵循最小必要原则，仅展示证明性标识（案例编号、图片 URL）；
- **Dify 依赖**：知识库检索依赖 Dify 服务可用；未配置时回退本地向量库（需要已构建索引）。

---

## 技术栈

- LangChain（结构化模型调用 / RAG Chain）
- LangGraph（状态图调度 / Skill 路由 / 事件流 / Checkpoint）
- Plan-Execute（多目标依赖计划）+ Skill 内有限纠错
- Streamlit（Web UI）
- Dify（知识库 / 向量检索）
- DashScope（text-embedding-v4 嵌入、qwen-vl-max 视觉）
- DeepSeek（对话模型，OpenAI 兼容接口）
- SQLite / Chroma / python-dotenv / pyyaml / requests
