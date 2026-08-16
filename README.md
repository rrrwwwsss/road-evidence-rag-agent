# 智扫通机器人智能客服

> 基于 LangChain ReAct Agent + RAG 的扫地机器人专业智能客服系统

---

## 目录

- [项目简介](#项目简介)
- [核心功能与架构](#核心功能与架构)
  - [整体架构](#整体架构)
  - [ReAct Agent](#react-agent)
  - [RAG 知识库检索](#rag-知识库检索)
  - [工具（Tools）](#工具tools)
  - [中间件（Middleware）](#中间件middleware)
  - [动态提示词切换](#动态提示词切换)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装依赖](#安装依赖)
  - [配置环境变量](#配置环境变量)
  - [初始化知识库](#初始化知识库)
  - [启动应用](#启动应用)
- [使用说明](#使用说明)
  - [Web 界面使用](#web-界面使用)
  - [直接调用 Agent](#直接调用-agent)
  - [示例问答](#示例问答)
- [配置说明](#配置说明)
- [技术栈](#技术栈)

---

## 项目简介

**智扫通机器人智能客服** 是一个专为扫地机器人和扫拖一体机器人用户打造的 AI 客服系统。系统具备自主的 ReAct（Reasoning + Acting）思考与工具调用能力，能够：

- 🤖 **专业问答**：回答扫地/扫拖机器人的选购建议、使用技巧、故障排查、维护保养等专业问题。
- 🌤️ **环境感知**：结合用户所在城市的实时天气，判断环境是否适合机器人工作。
- 📊 **使用报告**：自动获取用户 ID 与月份，查询历史使用记录，生成个性化的机器人使用情况报告与保养建议。
- 💬 **流式对话**：支持 Streamlit Web 界面与终端命令行两种交互方式，均以流式输出实现打字机效果。

---

## 核心功能与架构

### 整体架构

```
┌──────────────────────────────────────────┐
│          Streamlit Web UI (app.py)        │
└───────────────────┬──────────────────────┘
                    │ 用户输入（自然语言）
┌───────────────────▼──────────────────────┐
│          ReAct Agent (agent/)             │
│                                          │
│  系统提示词 ──► LLM ──► 工具调用决策      │
│                  ▲           │           │
│  中间件拦截 ◄────┘    Tools / Middleware  │
└───┬───────────────────────┬──────────────┘
    │                       │
    ▼                       ▼
┌───────────┐       ┌───────────────────┐
│ RAG 模块  │       │   外部工具调用     │
│ (rag/)    │       │                   │
│           │       │ • get_weather     │
│ Chroma    │       │ • get_user_id     │
│ 向量数据库 │       │ • get_user_location│
│           │       │ • fetch_external  │
│ 知识库文档 │       │   _data           │
└───────────┘       └───────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  知识库文档（data/）                       │
│  • 扫地机器人100问（PDF + TXT）            │
│  • 扫拖一体机器人100问                    │
│  • 故障排除指南                           │
│  • 维护保养手册                           │
│  • 选购指南                               │
└──────────────────────────────────────────┘
```

### ReAct Agent

Agent 采用 **ReAct（Reasoning + Acting）** 架构，严格遵循「思考 → 行动 → 观察 → 再思考」的循环流程：

1. **思考（Reasoning）**：分析用户问题，判断需要调用哪些工具、按什么顺序调用。
2. **行动（Acting）**：调用工具获取信息（天气、用户数据、知识库等）。
3. **观察（Observation）**：获取工具返回结果，判断信息是否足够回答问题。
4. **生成答案**：整合所有信息，生成流畅、专业的中文回答。

Agent 核心通过 `langchain.agents.create_agent` 构建，底层基于 **LangGraph** 运行。

### RAG 知识库检索

系统内置 **RAG（Retrieval-Augmented Generation）** 模块，将专业文档向量化后存储在 Chroma 数据库中：

| 知识库文件 | 内容描述 |
|------------|----------|
| `扫地机器人100问.pdf` | 扫地机器人常见问题与解答 |
| `扫地机器人100问2.txt` | 更多扫地机器人问答补充 |
| `扫拖一体机器人100问.txt` | 扫拖一体机器人专项问答 |
| `故障排除.txt` | 故障诊断与处理方案 |
| `维护保养.txt` | 日常维护与保养建议 |
| `选购指南.txt` | 机型选购建议与对比 |

- **文档分块**：`chunk_size=200`，`chunk_overlap=20`
- **Embedding 模型**：`text-embedding-v4`（阿里云 DashScope）
- **检索策略**：相似度检索，每次返回 Top-3 相关片段
- **去重机制**：基于 MD5 哈希，避免重复加载相同文档

### 工具（Tools）

Agent 配备了 7 个 LangChain 工具，覆盖从知识检索到用户数据获取的完整链路：

| 工具名称 | 参数 | 功能描述 |
|----------|------|----------|
| `rag_summarize` | `query: str` | 从向量知识库检索相关资料并总结回答 |
| `get_weather` | `city: str` | 获取指定城市的实时天气、湿度、降雨概率等信息 |
| `get_user_location` | 无 | 获取当前用户所在城市名称 |
| `get_user_id` | 无 | 获取当前用户的唯一 ID |
| `get_current_month` | 无 | 获取当前月份（`YYYY-MM` 格式） |
| `fetch_external_data` | `user_id: str, month: str` | 从外部系统获取指定用户在指定月份的使用记录 |
| `fill_context_for_report` | 无 | 触发中间件注入报告生成上下文，实现提示词动态切换 |

### 中间件（Middleware）

系统通过 3 个中间件对 Agent 执行过程进行全链路监控与干预：

| 中间件名称 | 触发时机 | 功能描述 |
|------------|----------|----------|
| `monitor_tool` | 每次工具调用前后 | 记录工具名称、入参、执行结果；当 `fill_context_for_report` 被调用时，将 `context["report"]` 标记置为 `True` |
| `log_before_model` | 每次调用 LLM 前 | 记录当前消息条数及最新消息内容，方便调试 |
| `report_prompt_switch` | 每次生成提示词时 | 根据运行时 `context["report"]` 标记动态切换系统提示词 |

### 动态提示词切换

系统内置 **3 套提示词**，在不同场景下自动切换：

```
context["report"] == False  →  使用 main_prompt.txt（通用客服提示词）
context["report"] == True   →  使用 report_prompt.txt（报告写手提示词）
RAG 总结链                  →  使用 rag_summarize.txt（RAG 总结专用提示词）
```

**报告生成的固定执行链路**（由系统提示词强约束）：

```
获取用户 ID → 获取当前月份 → fill_context_for_report → fetch_external_data → 生成报告
```

---

## 项目结构

```
AI_RAG_agent_project/
├── agent/
│   ├── react_agent.py          # ReactAgent 类，Agent 入口
│   └── tools/
│       ├── agent_tools.py      # 7 个 LangChain 工具定义
│       └── middleware.py       # 3 个中间件（监控、日志、提示词切换）
├── app.py                      # Streamlit Web 应用入口
├── config/
│   ├── agent.yml               # Agent 配置（外部数据路径等）
│   ├── chroma.yml              # Chroma 向量数据库配置
│   ├── prompts.yml             # 提示词文件路径配置
│   └── rag.yml                 # LLM 与 Embedding 模型配置
├── data/
│   ├── external/
│   │   └── records.csv         # 用户使用记录数据（10 用户，多月份）
│   ├── 扫地机器人100问.pdf
│   ├── 扫地机器人100问2.txt
│   ├── 扫拖一体机器人100问.txt
│   ├── 故障排除.txt
│   ├── 维护保养.txt
│   └── 选购指南.txt
├── model/
│   └── factory.py              # LLM 与 Embedding 模型工厂
├── prompts/
│   ├── main_prompt.txt         # 通用客服系统提示词
│   ├── rag_summarize.txt       # RAG 总结专用提示词
│   └── report_prompt.txt       # 报告生成专用提示词
├── rag/
│   ├── rag_service.py          # RAG 总结服务（检索 + 链式调用）
│   └── vector_store.py         # Chroma 向量库管理（加载、检索）
├── utils/
│   ├── config_handler.py       # YAML 配置加载器
│   ├── file_handler.py         # 文件读取与 MD5 去重工具
│   ├── logger_handler.py       # 日志配置（控制台 + 文件）
│   ├── path_tool.py            # 绝对路径工具
│   └── prompt_loader.py        # 提示词加载器
├── chroma_db/                  # Chroma 向量数据库持久化目录
├── logs/                       # 应用运行日志（按日期命名）
├── md5.text                    # 已处理文件的 MD5 记录（去重用）
└── .env                        # 环境变量（API Key 等，勿提交到 Git）
```

---

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云 DashScope API Key（用于 Embedding 模型 `text-embedding-v4`）
- OpenAI 兼容 API Key（用于 Chat 模型，支持 LongCat、OpenAI 等）

### 安装依赖

```bash
pip install langchain langchain-core langchain-community langchain-openai langchain-chroma
pip install langgraph streamlit
pip install python-dotenv pyyaml pypdf
```

### 配置环境变量

在项目根目录创建 `.env` 文件，填入以下内容：

```dotenv
# 阿里云 DashScope API Key（用于 Embedding 模型）
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Chat 模型的 API Key 与接口地址（支持 OpenAI 兼容格式）
OPENAI_API_KEY=your_openai_compatible_api_key_here
OPENAI_API_BASE=https://your-api-endpoint/openai
```

如需修改所使用的模型，编辑 `config/rag.yml`：

```yaml
# Chat 模型名称
chat_model_name: your-chat-model-name

# Embedding 模型名称
embedding_model_name: text-embedding-v4
```

### 初始化知识库

首次运行前，需将知识库文档向量化并存入 Chroma 数据库：

```bash
python rag/vector_store.py
```

> 系统会自动读取 `data/` 目录下所有 `.txt` 和 `.pdf` 文件，分块后写入 `chroma_db/`。基于 MD5 的去重机制可确保重复执行不会重复写入。

### 启动应用

**方式一：Streamlit Web 界面（推荐）**

```bash
streamlit run app.py
```

启动后在浏览器打开 `http://localhost:8501` 即可开始对话。

**方式二：命令行终端**

```bash
python agent/react_agent.py
```

> 修改 `react_agent.py` 末尾 `__main__` 块中的查询内容以测试不同问题。

---

## 使用说明

### Web 界面使用

1. 启动 Streamlit 应用后，页面顶部显示「**智扫通机器人智能客服**」标题。
2. 在底部输入框中输入问题，按 Enter 或点击发送。
3. 系统会显示「智能客服思考中...」，随后以打字机效果流式输出回答。
4. 对话历史会自动保留在当前会话中，支持多轮对话。

### 直接调用 Agent

```python
from agent.react_agent import ReactAgent

agent = ReactAgent()

# 流式输出（逐段打印）
for chunk in agent.execute_stream("小户型适合哪款扫地机器人？"):
    print(chunk, end="", flush=True)
```

### 示例问答

**🔍 专业知识问答**

```
用户：扫地机器人迷路了怎么办？
用户：高湿度环境适合用扫拖一体机器人吗？
用户：HEPA 滤网多久需要更换一次？
用户：我家有宠物，适合买哪款扫地机器人？
```

**🌤️ 天气与环境感知**

```
用户：今天深圳的天气适合用扫拖一体机器人吗？
用户：我现在所在的城市适合开启拖地功能吗？
```

**📊 个人使用报告**

```
用户：给我生成我的使用报告
用户：帮我查一下我的机器人上个月的使用情况
用户：生成我 2025 年 6 月的使用报告
```

> **报告生成流程说明**：当用户请求使用报告时，Agent 会严格按照以下步骤执行：
> 1. 调用 `get_user_id` 获取当前用户 ID
> 2. 调用 `get_current_month`（或使用用户指定月份）
> 3. 调用 `fill_context_for_report`（触发提示词自动切换为报告写手模式）
> 4. 调用 `fetch_external_data` 获取用户历史使用记录
> 5. 以 Markdown 格式生成包含使用情况与保养建议的专业报告

---

## 配置说明

### `config/chroma.yml`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `collection_name` | `agent` | Chroma 集合名称 |
| `persist_directory` | `chroma_db` | 向量库持久化目录 |
| `k` | `3` | 相似度检索返回的文档数量 |
| `data_path` | `data` | 知识库文档目录 |
| `chunk_size` | `200` | 文档分块大小（字符数） |
| `chunk_overlap` | `20` | 相邻分块的重叠字符数 |
| `allow_knowledge_file_type` | `[txt, pdf]` | 支持的知识库文件类型 |

### `config/rag.yml`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `chat_model_name` | `LongCat-Flash-Chat` | Chat 模型名称 |
| `embedding_model_name` | `text-embedding-v4` | Embedding 模型名称 |

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **Agent 框架** | LangChain、LangGraph |
| **Web UI** | Streamlit |
| **向量数据库** | Chroma (`langchain-chroma`) |
| **LLM** | OpenAI 兼容 Chat 模型（如 LongCat-Flash-Chat） |
| **Embedding** | 阿里云 DashScope `text-embedding-v4` |
| **文档处理** | PyPDFLoader、TextLoader、RecursiveCharacterTextSplitter |
| **配置管理** | PyYAML |
| **日志** | Python `logging`（控制台 + 文件双输出） |
| **环境变量** | python-dotenv |
| **去重** | Python `hashlib`（MD5） |
