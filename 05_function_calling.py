from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from constant import DEEPSEEK_API_KEY
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
)

@tool()
def plus_tool(a:float, b:float) -> float:
    """
    计算两数之和的工具
    :param a: 第一个相加的数
    :param b: 第二个相加的数
    :return: 返回两数相加后的结果
    """
    return a + b

@tool()
def sub_tool(a:float, b:float) -> float:
    """
    计算两数之和的工具
    :param a: 被减数
    :param b: 减数
    :return: 两束相减后的结果
    """
    return a - b

tools = {
    plus_tool.name: plus_tool,
    sub_tool.name: sub_tool,
}

# 大模型绑定工具
llm_with_tools = llm.bind_tools(tools=list(tools.values()))
message: AIMessage = llm_with_tools.invoke("计算111+5555")
# 如果返回值有tool_calls，说明选择了工具
# 注意：大模型指挥选择工具，而不会使用工具，需要手动执行得到结果，再把执行结果包装成ToolMessage类型再发给大模型
if message.tool_calls:
    tool_messages = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool = tools.get(tool_name)
        # 返回工具使用的结果
        result = tool.invoke(tool_args)
        print(f"工具执行后的结果{result}")

        # 将结果包装成ToolMessage
        tool_message = ToolMessage(
            content=str(result),
            name=tool_name,
            tool_call_id=tool_call["id"], # 必传，帮大模型定位传哪个工具
            artifact="一点备注信息，不会被传递给大模型",
        )
        tool_messages.append(tool_message)

    # 将工具执行结果再次发送给大模型
    response = llm_with_tools.invoke([
        HumanMessage(content="计算111+5555"),  # 原始用户消息
        message,  # 模型的初始响应(包含工具调用)
        *tool_messages  # 工具执行结果
    ])

    print("最终响应:", response.content)