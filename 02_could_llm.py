from constant import DEEPSEEK_API_KEY
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 调用API
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
)

# langchain_messages = [
#     SystemMessage("你是一个SEO专家"),
#     HumanMessage("给我科普一下SEO的基础知识")
# ]

message = llm.invoke("帮我谴责一下资本家")
print(message)