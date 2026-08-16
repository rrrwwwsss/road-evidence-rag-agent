# 非现场取证线索库智能查询系统

> 基于 LangChain ReAct Agent 的道路/现场取证智能查询系统，集成 **RAG 知识库（Dify）**、**SQL 案件数据库查询**、**Qwen-VL 图片违法识别** 与 **执法报告生成**。

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

系统面向道路/现场取证、证据检索与执法辅助场景，以 ReAct（Reasoning + Acting）Agent 为核心，根据用户问题自动路由到合适的工具：

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
                 ReAct Agent (agent/react_agent.py)
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
   rag_summarize          sql_query           vision_analyze
   (Dify 知识库检索)      (SQLite 查询)        (Qwen-VL 识别)
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               ▼
                  生成最终回答 / Markdown 报告
```

### ReAct Agent 与工具路由

Agent 由 `langchain.agents.create_agent` 组装，注册 4 个工具，并根据系统提示词（`prompts/main_prompt.txt`）自动路由：

| 问题类型 | 路由工具 | 说明 |
| --- | --- | --- |
| 计数/统计/按时间·支队·地点·违法类型查询 | `sql_query(question)` | 自然语言转只读 SQL，查 `data/wupin_tanwei_dabt.db` |
| 判定标准/法规依据/历史案例相似度 | `rag_summarize(query, doc_type="")` | Dify 检索；`doc_type` 可选 `standard`/`case` |
| 上传图片识别违法行为 | `vision_analyze(image, question, context="")` | Qwen-VL；`image` 为图片 ID 或 URL |
| 图片 + “是否属于道路养护”判定 | `rag_summarize` → `vision_analyze` | 强约束：先取 RAG 判定标准再交给视觉模型 |
| 生成/查询执法报告 | `sql_query` + `rag_summarize` + `fill_context_for_report` | 报告前必须调用 `fill_context_for_report` 切换报告提示词 |

### 中间件（Middleware）

- `monitor_tool`：记录每次工具调用与参数，并在调用 `fill_context_for_report` 后把运行时上下文 `report` 置为 `True`；
- `log_before_model`：模型调用前输出日志；
- `report_prompt_switch`：动态提示词切换，报告场景使用 `prompts/report_prompt.txt`，其余使用 `prompts/main_prompt.txt`。

### RAG 知识库（Dify）

- 检索走 Dify Dataset API：`POST /v1/datasets/{dataset_id}/retrieve`，支持 `semantic_search` / `full_text_search` / `hybrid_search`（混合检索需 Dify 配置 rerank 模型）；
- 支持按元数据 `doc_type`（`standard`/`case`）过滤；
- 未配置 Dify（`DIFY_API_KEY` 或 `dataset_id` 为空）时自动回退本地 Chroma 向量库。

### SQL 查询

- `rag/sql_service.py` 内部用大模型把自然语言问题转成只读 SQL（仅放行 `SELECT/WITH`），失败自动修正重试一次；
- 以只读模式（`mode=ro`）连接 SQLite，防止误写数据。

### 视觉识别

- `rag/vision_service.py` 调用 `qwen-vl-max`（DashScope），支持图片 ID / data URI / URL；
- 上传图片由前端注册为 `img_xxxx`，随会话持久化。

---

## 项目结构

```
├── app.py                        # Streamlit 入口（文字 + 图片上传）
├── agent/
│   ├── react_agent.py            # ReAct Agent 组装
│   └── tools/
│       ├── agent_tools.py        # 工具定义（rag_summarize / sql_query / vision_analyze / fill_context_for_report）
│       └── middleware.py         # 工具监控 / 日志 / 提示词切换
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
│   ├── main_prompt.txt           # 系统提示词（工具路由）
│   ├── rag_summarize.txt         # RAG 汇总提示词
│   └── report_prompt.txt         # 报告生成提示词
├── data/
│   ├── records.csv               # 案件原始数据
│   ├── wupin_tanwei_dabt.db      # 案件数据库（SQL 查询数据源）
│   ├── 违法案例库/*.md           # 知识库案例文档
│   ├── 道路养护判定标准.md       # 判定标准（Dify 中 doc_type=standard）
│   └── 公路范围内的合法物品判定标准.md
├── eval/                         # RAG 检索准确率评估
│   ├── eval_retrieval.py
│   └── test_set.jsonl
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

### 示例问题

- 计数统计：「擅自占用公路一共有多少条？」「2026年7月23日哪个支队最多？」
- 判定标准：「道路养护行为的判定标准是什么？」
- 历史案例：「与擅自占用、挖掘公路类似的历史案例有哪些？」
- 图片识别：上传图片后问「这张图片是什么违法行为？」
- 图片判定：上传图片后问「图中行为是否属于道路养护行为？」
- 报告生成：「请为2026年5月密云执法队的违法案件生成一份执法报告草稿」

---

## 配置说明

| 文件 | 说明 |
| --- | --- |
| `config/rag.yml` | `chat_model_name`（对话）、`embedding_model_name`（嵌入）、`vision_model_name`（视觉，如 `qwen-vl-max`） |
| `config/dify.yml` | Dify `api_base` / `dataset_id` / `search_method` / `top_k` / 阈值 |
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

---

## 注意事项与安全

- **密钥保护**：`.env` 已加入 `.gitignore`，请勿提交；历史版本中若出现过密钥请尽快轮换；
- **执法合规**：系统输出仅作为辅助参考，涉及法律定性的结论需「人工/执法部门复核」；
- **数据最小化**：报告与对外输出遵循最小必要原则，仅展示证明性标识（案例编号、图片 URL）；
- **Dify 依赖**：知识库检索依赖 Dify 服务可用；未配置时回退本地向量库（需要已构建索引）。

---

## 技术栈

- LangChain（ReAct Agent / Middleware / RAG Chain）
- Streamlit（Web UI）
- Dify（知识库 / 向量检索）
- DashScope（text-embedding-v4 嵌入、qwen-vl-max 视觉）
- DeepSeek（对话模型，OpenAI 兼容接口）
- SQLite / Chroma / python-dotenv / pyyaml / requests