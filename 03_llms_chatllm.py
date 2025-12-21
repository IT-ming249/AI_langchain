from constant import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# deepseek大概率会报错SSL证书问题，可以换别的模型再试试
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,  # DeepSeek API地址
    temperature=0.7,
    max_tokens=1024
)

messages = [
    SystemMessage(content="你是一个SEO专家"),
    HumanMessage(content="给我科普一下SEO的基础知识")
]

# 调用
response = llm.invoke(messages)
print(response.content)