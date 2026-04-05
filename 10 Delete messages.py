from langchain.messages import RemoveMessage
from langchain.agents import  create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime
from langgraph.types import Command

from llm import model
from util import extract_ai_response
@tool
def delete_message(keyword: str, runtime: ToolRuntime):
    """
        删除包含指定关键字的消息

        :arg keyword: 在消息中搜索的关键字
    """
    messages = runtime.state.get("messages")
    to_remove = []
    for message in messages:
        if keyword in message.content:
            to_remove.append(RemoveMessage(id=message.id))
    if len(to_remove) > 0:
        tool_message = ToolMessage(content="消息删除成功", tool_call_id=runtime.tool_call_id)
        return Command(
            update={"messages": to_remove + [tool_message]},
        )
    return "没有需要删除的消息"

if __name__ == '__main__':
    agent = create_agent(
        model=model,
        tools=[delete_message],
        checkpointer=InMemorySaver()
    )
    config = RunnableConfig(configurable={"thread_id": "1"})

    response = agent.invoke({
        "messages": [
            HumanMessage(content="请写一条关于小狗的打油诗"),
        ]
    }, config=config)
    # print(response)

    response_move = agent.invoke({
        "messages": [
            HumanMessage(content="请将所有包含打油诗的消息都删掉")
        ]
    }, config=config)

    print(response_move)


