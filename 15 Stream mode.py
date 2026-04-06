from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from llm import model
import asyncio


def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"{city} 今天晴，气温 20-25℃，适合出门。"

# update模式适合调试
async def stream_agent_updates(agent):
    async for chunck in agent.astream(
        {"messages":[HumanMessage(content="深圳天气怎么样？")]},
        stream_model="updates"
    ):
        # chunck是字典类型，key为节点名称
        for step, data in chunck.items():
            print(f"当前步骤{step}\n")
            if "messages" in data:
                for message in data["messages"]:
                    message.pretty_print()

# messages适合实时输出
async def stream_agent_messages():
    async for step in agent.astream(
        {"messages": [HumanMessage(content="写一首关于春天的诗")]},
        stream_mode="messages",
    ):
        # step 是一个元组：(chunk, metadata)
        chunk, metadata = step
        print(chunk.content, end="", flush=True)


@tool
def long_running_task(query: str, runtime: ToolRuntime) -> str:
    """执行一个耗时的任务，并实时报告进度"""
    writer = runtime.stream_writer

    writer("开始处理查询...")
    # 模拟耗时操作
    import time
    time.sleep(0.5)

    writer("正在检索数据库...")
    time.sleep(0.5)

    writer("正在生成结果...")
    time.sleep(0.5)

    return f"查询结果：关于 '{query}' 的信息"

# custom模型自定义输出
async def stream_custom(agent):
    async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": "查询 Python 教程"}]},
            stream_mode="custom", # 可以同时指定多种模式
    ):
        print(f"[自定义更新] {chunk}")

async def stream_mix_modes(agent):
    async for stream_mode, data in agent.astream(
        {"messages": [{"role": "user", "content": "帮我查找一下Python相关的信息"}]},
        stream_mode=["messages", "updates", "custom"],  # 同时订阅 Token 与步骤完成
    ):
        if stream_mode == "messages":
            chunk, metadata = data
            if chunk.content:
                print(chunk.content, end="", flush=True)
        elif stream_mode == "updates":
            # 每个步骤完成时打印一次
            for step, update in data.items():
                last_msg = update["messages"][-1]
                print(f"\n[步骤完成] {step}: {last_msg.content}")
        else:
            print(f"[自定义消息]{data}")


if __name__ == '__main__':

    agent = create_agent(
        model=model,
        tools=[get_weather, long_running_task],
        system_prompt="你是一个私人助手，可以调用工作为用户提供帮助。"
    )
    # asyncio.run(stream_agent_updates(agent))
    # asyncio.run(stream_custom(agent))
    asyncio.run(stream_mix_modes(agent))
