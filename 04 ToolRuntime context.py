from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from llm import model
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


@dataclass
class UserContext:
    user_id: str

# 相当于↓
# class UserContext:
#     def __init__(self, user_id:  str):
#         self.user_id = user_id

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前用户的账户信息。"""
    user_id = runtime.context.user_id

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
        context_schema=UserContext,
        system_prompt="你是一个银行客服，可以调用已有的工具获取用户账户信息。"
    )
    # context_schema 可以是TypeDict, dataclass 和 BaseModel类型
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我现在账户里有多少钱？"}]},
        context=UserContext(user_id="user123"),
    )
    print(extract_ai_response( result))


if __name__ == '__main__':
    main()
