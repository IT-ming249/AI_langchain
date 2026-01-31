from llm import model
from util import extract_ai_response
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool

@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]
    # State用于存储一些可变数据，比如消息、计数器等。
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"


def main():
    agent = create_agent(
        model=model,
        tools=[summarize_conversation],
        system_prompt="你是一个个人情感助手，当用户提示总结消息时，调用工具summarize_conversation总结消息。"
    )
    response = agent.invoke({
        "messages": [
            HumanMessage(content="你好，我是张三，请问你是？"),
            HumanMessage(content="请讲一个冷笑话。")
        ],
    })
    messages = list(response["messages"])
    messages.append(HumanMessage("请总结下消息"))
    response1 = agent.invoke({"messages": messages})
    print(extract_ai_response( response1))

if __name__ == '__main__':
    main()