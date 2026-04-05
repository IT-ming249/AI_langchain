from typing import Dict

from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel
from llm import model
from langgraph.checkpoint.memory import InMemorySaver


class CustomState(AgentState):
    user_name: str


class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    user_id: str,
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command | str:
    """查询并更新用户信息到短期记忆中。"""
    # user_id = runtime.context.user_id
    name = "张三" if user_id == "user_123" else "Unknown user"

    if name == "Unknown user":
        return "未知用户"

    return Command(
        update={
            "user_name": name,  # 写入自定义状态字段
            "messages": [
                ToolMessage(
                    "用户信息更新成功",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

@tool
def get_user_info(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """根据已经写入 memory 的 user_name 打招呼。"""
    # 这里面是直接从state里面拿，说明工具调用了短期记忆
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        # 如果还没查到，就提示模型先去调用另一个工具
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "请先调用'update_user_info'工具,用以更新用户信息",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    return f"你好，{user_name}!"

if __name__ == '__main__':
    agent = create_agent(
        model=model,
        tools=[update_user_info, get_user_info],
        state_schema=CustomState,
        context_schema=CustomContext,
        checkpointer=InMemorySaver()
    )
    config = RunnableConfig(configurable={"thread_id": "111"})
    agent.invoke(
        {"messages": [{"role": "user", "content": "我的用户ID是：user_123，更新下我的信息即可，不要获取信息。"}]},
        config=config
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "我的用户名是什么？"}]},
        config=config
    )
    print(response)