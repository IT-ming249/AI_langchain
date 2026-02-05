from typing import Any
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore
from llm import model
from util import extract_ai_response


@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """根据用户 ID 查询用户信息。"""
    store = runtime.store
    # 指定命名空间users, key为user_id
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """保存用户信息到 Store 中。"""
    store = runtime.store
    # 存储时的命名空间必须跟获取时的命名空间一致
    store.put(("users",), user_id, user_info)
    return "用户信息已保存。"



def main():
    # 这个store可以时数据库对象，此处使用内存存储做演示
    store = InMemoryStore()

    agent = create_agent(
        model,
        tools=[get_user_info, save_user_info],
        store=store,
    )

    # 第一次对话：保存用户信息
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "保存一个用户：id=abc123, name=Foo, age=25, "
                        "email=[email protected]"
                    ),
                }
            ]
        }
    )

    # 第二次对话：读取用户信息
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "查询 id 为 abc123 的用户信息。",
                }
            ]
        }
    )
    print(extract_ai_response(result))

if __name__ == '__main__':
    main()