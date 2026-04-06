from langchain.agents import create_agent
from langchain.messages import HumanMessage

from llm import model
import asyncio
from langchain.tools import tool


"""
astream_events 会生成多种类型的事件：
● on_chat_model_start：聊天模型开始执行
● on_chat_model_stream：聊天模型正在输出 Token
● on_chat_model_end：聊天模型执行完成
● on_tool_start：工具调用开始
● on_tool_end：工具调用结束
● on_chain_start：链开始执行
● on_chain_end：链执行结束
● on_parser_start：解析器开始工作
● on_parser_end：解析器完成工作

每个事件都是一个字典，包含以下字段：
{
    "event": "on_chat_model_stream",  # 事件类型
    "name": "ChatOpenAI",              # 组件名称
    "run_id": "abc123",                # 运行 ID
    "tags": [],                        # 标签
    "metadata": {},                    # 元数据
    "data": {                          # 事件数据
        "chunk": AIMessageChunk(...)
    }
}
可以通过 event["name"] 来区分不同组件的事件，通过 event["run_id"] 来追踪同一个执行流程中的多个事件。
"""

@tool
def get_weather(city: str) -> str:
    """根据城市获取天气信息"""
    return f"{city} 今天晴，气温 20-25℃，适合出门。"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个天气助手，可以查询天气信息。",
)

async def stream_events():
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="请问明天上海的天气怎么样？")]},
    ):
        kind = event["event"]

        # 筛选模型流式输出事件
        if kind == "on_chain_start":
            print("链开始执行")
            print(event)
            print()
        elif kind == "on_chain_end":
            print("链结束执行")
            print(event)
            print()
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_tool_start":
            print("开始调用工具")
            print(event)
            print()
        elif kind == "on_tool_end":
            print("工具调用结束")
            print(event)
            print()
        elif kind == "on_chat_model_start":
            print("模型开始思考...")
            print(event)
            print()
        elif kind == "on_chat_model_end":
            print("模型完成思考...")
            print(event)
            print()


asyncio.run(stream_events())