# app.py
from dotenv import load_dotenv
load_dotenv()  # 👈 必须在所有其他 import 之前！
import asyncio
import base64
import uuid

import streamlit as st
from agent.display_labels import localize_trace
from agent.react_agent import ReactAgent


st.set_page_config(
    page_title="非现场取证线索库 · 智能查询",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_app_styles() -> None:
    """为取证查询工作台注入轻量、业务化的视觉样式。"""
    st.markdown(
        """
        <style>
        :root {
            --forensic-navy: #102A43;
            --forensic-blue: #1769AA;
            --forensic-cyan: #159EAF;
            --forensic-ink: #243B53;
            --forensic-muted: #6B7C93;
            --forensic-line: #DCE6EF;
            --forensic-surface: #F6F9FC;
        }

        .stApp {
            background:
                radial-gradient(circle at 82% 2%, rgba(21, 158, 175, .08), transparent 27rem),
                linear-gradient(180deg, #F8FBFD 0%, #FFFFFF 34%);
            color: var(--forensic-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            padding-top: 1.6rem;
            padding-bottom: 7rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D2740 0%, #123B59 58%, #0E4D5C 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color: #EAF3F8; }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 2.75rem;
            border: 1px solid rgba(255,255,255,.16);
            background: rgba(255,255,255,.08);
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #49C3CF;
            background: rgba(73,195,207,.14);
        }
        .brand-lockup { padding: .35rem .15rem 1.25rem; }
        .brand-mark {
            display: inline-grid; place-items: center; width: 42px; height: 42px;
            border-radius: 12px; background: linear-gradient(135deg, #2BB8C5, #2B78C5);
            box-shadow: 0 8px 22px rgba(0,0,0,.2); font-size: 21px; margin-bottom: .8rem;
        }
        .brand-title { color: white; font-weight: 750; font-size: 1.05rem; letter-spacing: .02em; }
        .brand-subtitle { color: #AFC5D5; font-size: .76rem; margin-top: .3rem; letter-spacing: .08em; }
        .side-section { color: #8EB1C5; font-size: .7rem; letter-spacing: .12em; margin: 1.1rem 0 .55rem; }
        .side-status {
            padding: .8rem .9rem; border-radius: 12px; background: rgba(255,255,255,.06);
            border: 1px solid rgba(255,255,255,.1); font-size: .8rem;
        }
        .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#5CE1A5; margin-right:.5rem; box-shadow:0 0 0 4px rgba(92,225,165,.12); }

        .case-header {
            position: relative; overflow: hidden; padding: 1.4rem 1.55rem;
            border: 1px solid var(--forensic-line); border-radius: 18px;
            background: linear-gradient(115deg, rgba(255,255,255,.98), rgba(237,248,251,.94));
            box-shadow: 0 14px 40px rgba(27, 64, 92, .08); margin-bottom: 1.25rem;
        }
        .case-header:after {
            content:""; position:absolute; right:-35px; top:-48px; width:190px; height:190px;
            border: 24px solid rgba(23,105,170,.055); border-radius:50%;
        }
        .case-kicker { color: var(--forensic-blue); font-size: .72rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .35rem; }
        .case-title { color: var(--forensic-navy); font-size: 1.62rem; line-height: 1.25; font-weight: 780; margin: 0; }
        .case-desc { color: var(--forensic-muted); font-size: .88rem; margin-top: .45rem; }
        .case-badges { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.9rem; }
        .case-badge { padding:.28rem .58rem; border-radius:999px; border:1px solid #CFE1ED; background:#F7FBFD; color:#42657A; font-size:.72rem; }

        .welcome { text-align:center; padding: 2.2rem .5rem 1rem; }
        .welcome-icon {
            display:inline-grid; place-items:center; width:58px; height:58px; border-radius:17px;
            background:linear-gradient(135deg,#E4F3FA,#DCF7F3); color:#1769AA; font-size:27px;
            border:1px solid #D3E8EF; box-shadow:0 10px 28px rgba(24,94,122,.09);
        }
        .welcome h2 { color:var(--forensic-navy); font-size:1.25rem; margin:.85rem 0 .35rem; }
        .welcome p { color:var(--forensic-muted); font-size:.87rem; margin:0; }

        [data-testid="stChatMessage"] {
            border: 1px solid #E1EAF1; border-radius: 15px; padding: .55rem .7rem;
            background: rgba(255,255,255,.9); box-shadow: 0 5px 18px rgba(26,65,91,.045);
            margin-bottom: .8rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg,#F0F7FC,#F5FAFD); border-color:#D6E6F1;
        }
        [data-testid="stChatInput"] {
            border: 1px solid #C9D9E5; box-shadow: 0 12px 32px rgba(18,64,91,.12); border-radius: 16px;
            background: rgba(255,255,255,.97);
        }
        [data-testid="stStatusWidget"] { border-radius: 12px; border-color: #DCE6EF; }
        div[data-testid="stExpander"] { border-color:#E0E8EF; border-radius:10px; background:#FAFCFD; }
        .stButton > button {
            border-radius: 11px; border-color:#D8E4EC; color:#274C63; background:#FFFFFF;
            min-height: 3.25rem; text-align:left; transition: all .16s ease;
        }
        .stButton > button:hover { border-color:#159EAF; color:#0D7180; transform:translateY(-1px); box-shadow:0 7px 18px rgba(21,112,128,.09); }
        hr { border-color:#E2EAF0 !important; }

        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] { padding-top: .75rem; }
            .case-header { padding: 1.05rem; border-radius: 14px; }
            .case-title { font-size: 1.3rem; }
            .case-desc { font-size: .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_app_styles()


AGENT_SCHEMA_VERSION = 7

STATUS_ICONS = {
    "running": ":material/progress_activity:",
    "complete": ":material/check_circle:",
    "error": ":material/error:",
    "skipped": ":material/skip_next:",
}


def render_audit_event(item: dict) -> None:
    """渲染一条可点击展开的审计事件，不展示模型内部思维链。"""
    title = localize_trace(item.get("title", "处理步骤"))
    icon = STATUS_ICONS.get(item.get("status", "running"), ":material/info:")
    show_reasoning = item.get("event") in {"reasoning_summary", "analysis_summary"}
    with st.expander(f"{icon} {title}", expanded=show_reasoning):
        st.caption(
            f"{item.get('category', '系统处理')} · "
            f"{item.get('created_at', '')} · 事件 {item.get('sequence', '-') }"
        )
        st.write(item.get("explanation", "系统执行过程中的可审计事件。"))
        details = item.get("details") or {}
        if details.get("detail_type") == "reasoning":
            st.markdown("**推理摘要（可审计）**")
            st.write(details.get("summary", ""))
            reasoning_type = details.get("reasoning_type")
            if reasoning_type == "intent":
                st.markdown(
                    f"**主意图：** {details.get('primary_intent', '-')}  "
                    f"**置信度：** {details.get('confidence', '-')}"
                )
                if details.get("secondary_intents"):
                    st.write("附加意图：" + "、".join(details["secondary_intents"]))
                if details.get("extracted_entities"):
                    st.markdown("**提取的查询条件**")
                    st.json(details["extracted_entities"], expanded=True)
                if details.get("goals"):
                    st.markdown("**识别到的业务目标**")
                    st.dataframe(details["goals"], width="stretch", hide_index=True)
            elif reasoning_type == "plan":
                st.caption(
                    f"执行模式：{details.get('execution_mode', 'Plan-Execute')} · "
                    f"单个 Skill 最大迭代：{details.get('max_skill_iterations', '-')}"
                )
                if details.get("steps"):
                    st.dataframe(details["steps"], width="stretch", hide_index=True)
            elif reasoning_type == "conclusion":
                st.markdown(
                    f"**执行状态：** {details.get('execution_status', '-')}  "
                    f"**工具结果：** {details.get('tool_result_count', 0)} 个  "
                    f"**证据：** {details.get('evidence_count', 0)} 条"
                )
                if details.get("selected_skills"):
                    st.write("使用的业务能力：" + " → ".join(details["selected_skills"]))
                if details.get("sources"):
                    st.markdown("**结论引用来源**")
                    st.dataframe(details["sources"], width="stretch", hide_index=True)
                for warning in details.get("warnings", []):
                    st.warning(warning)
                for error in details.get("errors", []):
                    st.error(error)
        elif details:
            st.markdown(
                f"**调用工具：** {details.get('tool_label', '未知工具')} "
                f"(`{details.get('tool_name', '-')}`)  "
                f"**耗时：** {details.get('latency_ms', 0)} ms"
            )
            if details.get("error"):
                st.error(details["error"])
            if details.get("request"):
                st.markdown("**调用参数**")
                st.json(details["request"], expanded=True)
            if details.get("command"):
                st.markdown("**执行命令**")
                st.code(details["command"], language="sql")

            result = details.get("result") or {}
            rows = result.get("rows_preview")
            if rows is not None:
                st.markdown(
                    f"**查询结果：** 共 {result.get('row_count', len(rows))} 行，"
                    f"显示前 {result.get('preview_limit', 10)} 行"
                )
                if rows:
                    st.dataframe(rows, width="stretch", hide_index=True)
                else:
                    st.caption("查询成功，但没有返回匹配数据。")
            if result.get("summary"):
                st.markdown("**检索摘要**")
                st.write(result["summary"])
            if result.get("analysis"):
                st.markdown("**图片分析结果**")
                st.write(result["analysis"])
            if result.get("upstream_error"):
                st.warning(f"上游服务异常：{result['upstream_error']}")

            sources = details.get("sources") or []
            if sources:
                source_total = result.get("source_count", len(sources))
                st.markdown(
                    f"**命中来源：** 共 {source_total} 条，当前展示 {len(sources)} 条"
                )
                st.dataframe(sources, width="stretch", hide_index=True)

# 集中初始化每个浏览器会话的状态；LangGraph thread_id 在页面重跑时保持稳定。
st.session_state.setdefault("message", [])
st.session_state.setdefault("uploaded_images", {})
st.session_state.setdefault("agent_thread_id", "chat_" + uuid.uuid4().hex)


def reset_conversation() -> None:
    """开启新的查询会话，同时清理本会话上传的临时图片。"""
    st.session_state["message"] = []
    st.session_state["uploaded_images"] = {}
    st.session_state["agent_thread_id"] = "chat_" + uuid.uuid4().hex
    st.session_state.pop("agent", None)
    st.session_state.pop("agent_schema_version", None)


with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">⌕</div>
          <div class="brand-title">非现场取证线索库</div>
          <div class="brand-subtitle">INTELLIGENT EVIDENCE SEARCH</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("＋  新建查询", on_click=reset_conversation, use_container_width=True)
    st.markdown('<div class="side-section">工作台</div>', unsafe_allow_html=True)
    st.markdown("**▣  智能线索查询**")
    st.caption("自然语言检索 · 多源证据关联")
    st.markdown('<div class="side-section">本次会话</div>', unsafe_allow_html=True)
    conversation_rounds = sum(
        1 for item in st.session_state["message"] if item.get("role") == "user"
    )
    uploaded_count = len(st.session_state["uploaded_images"])
    metric_left, metric_right = st.columns(2)
    metric_left.metric("查询轮次", conversation_rounds)
    metric_right.metric("上传图片", uploaded_count)
    st.markdown('<div class="side-section">系统状态</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-status"><span class="status-dot"></span>'
        '线索检索服务已就绪<br><span style="color:#92ADBD;font-size:.7rem;">'
        '结果仅供执法研判参考</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="case-header">
      <div class="case-kicker">EVIDENCE INTELLIGENCE · 线索研判工作台</div>
      <h1 class="case-title">非现场取证线索库智能查询系统</h1>
      <div class="case-desc">面向案件线索的一站式检索、图片研判与证据溯源，帮助快速定位有效信息。</div>
      <div class="case-badges">
        <span class="case-badge">结构化案件查询</span>
        <span class="case-badge">知识库检索</span>
        <span class="case-badge">图像线索分析</span>
        <span class="case-badge">全流程可审计</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# 热更新后旧类实例仍会留在 session_state；协议升级时必须重建 Agent。
if (
    "agent" not in st.session_state
    or st.session_state.get("agent_schema_version") != AGENT_SCHEMA_VERSION
):
    restored_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state["message"]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    st.session_state["agent"] = ReactAgent(
        thread_id=st.session_state["agent_thread_id"],
        history=restored_history,
    )
    st.session_state["agent_schema_version"] = AGENT_SCHEMA_VERSION

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        trace = message.get("trace", [])
        audit_events = message.get("audit_events", [])
        if message["role"] == "assistant" and (audit_events or trace):
            event_count = len(audit_events) or len(trace)
            trace_state = "complete" if message.get("process_ok", True) else "error"
            with st.status(
                f"分析与执行过程（{event_count} 步，可点击查看）",
                state=trace_state,
                expanded=False,
                type="compact",
            ):
                if audit_events:
                    for item in audit_events:
                        render_audit_event(item)
                else:
                    for item in trace:
                        st.write(localize_trace(item))
        st.write(message["content"])

starter_prompt = None
if not st.session_state["message"]:
    st.markdown(
        """
        <div class="welcome">
          <div class="welcome-icon">⌕</div>
          <h2>从一条线索开始研判</h2>
          <p>描述案件要素、时间地点或上传现场图片，系统将自动选择检索与分析能力。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quick_cols = st.columns(3)
    with quick_cols[0]:
        if st.button("📋  查询近期案件\n\n按时间、地点或类型筛选", use_container_width=True):
            starter_prompt = "查询近期的案件线索，并按时间倒序展示关键信息。"
    with quick_cols[1]:
        if st.button("🔗  关联相似线索\n\n发现案件之间的潜在联系", use_container_width=True):
            starter_prompt = "请帮我检索并分析相似或可能关联的案件线索。"
    with quick_cols[2]:
        if st.button("🖼️  研判现场图片\n\n上传图片识别关键取证要素", use_container_width=True):
            st.info("请点击下方输入框左侧的附件按钮上传现场图片，并补充需要研判的问题。")

# 用户输入提示词（支持上传图片）
prompt = st.chat_input(
    "输入案件要素、查询条件，或上传现场图片…",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
    submit_mode="disable",
)

if starter_prompt:
    prompt = starter_prompt

if prompt:
    # accept_file=True 时 prompt 为 ChatInputValue（类 dict，含 text 与 files）
    if isinstance(prompt, str):
        text = prompt
        files = []
    else:
        text = prompt["text"]
        files = prompt["files"]

    image_marks = []
    for uploaded_file in files:
        img_id = "img_" + uuid.uuid4().hex[:10]
        data_uri = "data:{};base64,{}".format(
            uploaded_file.type or "image/jpeg",
            base64.b64encode(uploaded_file.getvalue()).decode("utf-8"),
        )
        st.session_state["uploaded_images"][img_id] = data_uri
        image_marks.append(f"【上传图片：图片ID {img_id}】")

    if image_marks:
        text = (text + "\n" + "\n".join(image_marks)).strip()

    if not text:
        st.stop()

    with st.chat_message("user"):
        st.write(text)
        for uploaded_file in files:
            st.image(uploaded_file.getvalue(), width=320)

    st.session_state["message"].append({"role": "user", "content": text})

    audit_events: list[dict] = []
    with st.chat_message("assistant"):
        process_status = st.status(
            "正在分析并执行请求…",
            expanded=True,
            type="compact",
        )

        async def consume_agent_events() -> tuple[str, bool]:
            final_answer = "系统未生成有效回答。"
            process_ok = False
            async for stream_event in st.session_state["agent"].astream_events(
                text,
                version="v2",
                uploaded_images=dict(st.session_state["uploaded_images"]),
            ):
                if stream_event["event"] == "on_custom_event":
                    item = stream_event["data"]
                    audit_events.append(item)
                    render_audit_event(item)
                    process_status.update(
                        label=f"正在分析并执行请求…（{len(audit_events)} 步）",
                        state="running",
                        expanded=True,
                    )
                elif stream_event["event"] == "on_chain_end":
                    final_answer = stream_event["data"]["output"]
                    process_ok = stream_event["data"]["status"] == "completed"
            return final_answer, process_ok

        final_response, process_ok = asyncio.run(consume_agent_events())
        process_status.update(
            label=(f"分析与执行完成（{len(audit_events)} 步，可点击查看）" if process_ok
                   else f"分析与执行未完成（{len(audit_events)} 步，可点击查看）"),
            state="complete" if process_ok else "error",
            expanded=not process_ok,
        )
        st.write(final_response)

    st.session_state["message"].append({
        "role": "assistant",
        "content": final_response,
        "audit_events": audit_events,
        "process_ok": process_ok,
    })
    st.rerun()
