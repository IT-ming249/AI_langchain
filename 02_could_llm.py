from constant import DEEPSEEK_API_KEY
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 调用API
llm = ChatDeepSeek(
    model="deepseek-chat",  # 或者 "deepseek-reasoner"
    api_key=DEEPSEEK_API_KEY,
    temperature=0.2,  # 新增：控制随机性
    max_tokens=2048,  # 新增：控制最大生成token数
    # timeout=30,  # 可选：超时设置
    # base_url="https://api.deepseek.com",  # 可选：自定义API端点
)

# 方式1：直接调用（简单查询）
message = llm.invoke("帮我谴责一下资本家")
print(message.content)  # 重要：使用 .content 获取文本内容

# # 方式2：使用消息格式（推荐，支持对话历史）
# print("\n" + "="*50 + "\n")
#
# messages = [
#     SystemMessage(content="你是一个幽默的马克思主义经济学家"),
#     HumanMessage(content="帮我谴责一下资本家")
#
#
# response = llm.invoke(messages)
# print(f"回答: {response.content}")

# 方式3：流式输出（新增功能）
# print("\n" + "="*50 + "\n")
# print("流式输出示例:")
#
# for chunk in llm.stream("帮我谴责一下资本家"):
#     print(chunk.content, end="", flush=True)