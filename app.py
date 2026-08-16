# app.py
from dotenv import load_dotenv
load_dotenv()  # 👈 必须在所有其他 import 之前！
import base64
import time
import uuid

import streamlit as st
from agent.react_agent import ReactAgent
from rag.vision_service import register_uploaded_image, _uploaded_images

# 标题
st.title("非现场取证线索库智能查询系统")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = {}

# 每次运行都把会话内已上传的图片重新注册进工具进程（防止历史轮次/重跑后图片ID失效）
for img_id, data_uri in st.session_state["uploaded_images"].items():
    if img_id not in _uploaded_images:
        register_uploaded_image(img_id, data_uri)

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词（支持上传图片）
prompt = st.chat_input(
    "请输入查询问题，或上传图片后提问（支持 jpg/png）",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)

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
        register_uploaded_image(img_id, data_uri)
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

    response_messages = []
    with st.spinner("智能助理思考中..."):
        res_stream = st.session_state["agent"].execute_stream(text)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()