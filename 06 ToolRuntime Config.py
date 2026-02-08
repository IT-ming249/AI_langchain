from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from llm import model
from langchain.messages import HumanMessage
from util import extract_ai_response


USER_DATABASE = {
    "user123": {
        "name": "Alice",
        "account_type": "Premium",
        "balance": 5000,
        "email": "[email protected]",
    },
    "user456": {
        "name": "Bob",
        "account_type": "Standard",
        "balance": 1200,
        "email": "[email protected]",
    },
}

@tool
def get_account_info(
    runtime: ToolRuntime
):
    """获取当前用户的账户信息。"""
    user_id = runtime.config['configurable'].get('user_id')

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (
            f"账户名：{user['name']}\n"
            f"账户类型：{user['account_type']}\n"
            f"余额：${user['balance']}"
        )
    return "未找到该用户。"

def main():
    agent = create_agent(
        model=model,
        tools=[get_account_info],
        system_prompt="你是一个金融助理。"
    )
    response = agent.invoke(
        {"messages": [HumanMessage(content="我现在账户上有多少余额？")]},
        # tags, max_concurrency 这些有限定用途的参数必须用config进行传递
        # 自定义参数可以用context或者config配configurable参数进行传递，但context从语义和校验上都更强点，不易出错
        config={"configurable": {"user_id": "user456"}},
    )
    print(extract_ai_response(response))

if __name__ == '__main__':
    main()