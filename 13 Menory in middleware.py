from typing import TypedDict

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, after_model
from langchain.tools import tool
from langchain_core.messages import RemoveMessage
from langgraph.runtime import Runtime

from llm import model

class CustomContext(TypedDict):
    user_name: str
@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"{city}明天的天气是零下5摄氏度，会下雪。"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"你是一个得力个人助手，你现在服务的对象为：{user_name}, 每次回复消息都要带称呼和敬语"
    return system_prompt

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """如果模型回复中包含敏感词，就删掉该条消息。"""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]

    # 这里假设 content 是字符串，真实场景下要考虑多模态消息结构
    if any(word in str(last_message.content) for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

# 注意state 里绝对拿不到 context 上下文只有 runtime 才能拿到 context

if __name__ == '__main__':
    agent = create_agent(
        model=model,
        tools=[get_weather],
        context_schema=CustomContext,
        middleware=[dynamic_system_prompt, validate_response]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京明天的天气如何？"}]},
        context=CustomContext(user_name="张三"),
    )

    for msg in result["messages"]:
        msg.pretty_print()