from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# 加载大模型
llm = ChatOllama(model="qwen3:1.7b")

## 调用大模型的消息格式

# 1. 字典类型格式
dict_messages = [
    {'role': "system", "content": "你是一个SEO专家"},
    {"role": "user", "content": "给我简单科普一下SEO"},
]

# 2. 元组格式
tuple_messages = [
    ("system", "你是一个SEO专家"),
    ("user", "给我简单科普一下SEO")
]

# 3. langchain封装好的消息格式
langchain_messages = [
    SystemMessage("你是一个SEO专家"),
    HumanMessage("给我科普一下SEO的基础知识")
]
# 整个输出
# print(llm.invoke(langchain_messages))

# 流式输出
for chunk in llm.stream(input=langchain_messages):
    print(chunk.content, end="")