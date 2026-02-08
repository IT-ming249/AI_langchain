from llm import model
from tool_shema import WeatherInput
from util import extract_ai_response

from langchain.agents import  create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

# 工具的输入/输出类型 一定要指定， docstring一定要写
@tool
def search_knowledge_base(query: str) -> str:
    """在内部知识库中搜索答案（此处用模拟数据代替）。

    Args:
        query: 用户想要查询的问题或关键词。
    """
    return "请假流程：首先需要向部门leader最少提前三天发起请求需求（急事除外），部门leader同意后即可。"

# 入参较多且复杂的情况下
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """获取当前天气信息（示例函数，真实项目中应调用天气 API）字段名一定要跟args_schema指定的schema 一样"""
    temp = 22 if units == "celsius" else 72
    try:
        result = f"当前 {location} 天气：{temp} 度（单位：{units}）"
        if include_forecast:
            result += "\n未来 5 天预报：晴为主"
        return result
    except Exception as e:
        # 对异常进行捕获并返回“可读错误信息”，避免直接抛出异常导致整次调用失败。
        return f'获取天气数据异常{e}'


def main():
    agent = create_agent(
        model=model,
        tools=[search_knowledge_base, get_weather],
        system_prompt="你是公司内部知识库助手，可以调用已有的工具回答用户问题。"
    )

    res = agent.invoke({
        "messages": [HumanMessage(content="公司请假流程是怎么样的")]
    })
    print(extract_ai_response(res))


if __name__ == '__main__':
    # 获取工具名称
    # print(search_knowledge_base.name)
    # 获取工具描述信息
    # print(search_knowledge_base.description)
    main()