import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # 这会自动加载 .env 中的环境变量
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, sql_query, vision_analyze,
                                     fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

class ReactAgent:
    def __init__(self):
        # create_agent会把基础聊天模型、系统提示词、工具列表和中间件封装成一个智能体，负责决策何时调用工具并调用 LLM 生成响应。
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            # agent调取工具就是在方法上面加个@tool，然后在create_agent函数的tools变量里传方法名
            tools=[rag_summarize, sql_query, vision_analyze, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("请统计2026年5月密云执法队的案件情况并生成执法报告草稿"):
        print(chunk, end="", flush=True)