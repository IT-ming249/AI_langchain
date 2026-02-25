from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # 内存型 checkpointer
from llm import model
from util import extract_ai_response


"""关键点：
● 短期记忆是通过 checkpointer + thread_id 实现的；
● 同一线程内的多次 invoke 会共享 messages 等状态；
● 换一个 thread_id 就相当于“开了一个新会话窗口”。
"""

def get_user_info(name: str) -> str:
    """一个非常简单的工具函数，占位用"""
    return f"已记录用户名称：{name}"


agent = create_agent(
    model=model,
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  # 开启短期记忆（线程级）
)

# 为当前会话指定一个 thread_id
config = {"configurable": {"thread_id": "demo-thread-1"}}


agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    config=config,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Do you known my name?"}]},
    config=config,
)

print(extract_ai_response(result))



