from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# 1. LLMs (只支持一问一答， 纯文本)
# llm = ChatOllama(model="qwen3:1.7b")
# message = llm.invoke("我是一个开发牛马，说点让我开心的事情")
# print(message)  #返回字符串类型

# 2. ChantModel 支持多轮对话，结构化输出，多模态输入输出， Function calling
llm = ChatOllama(model="qwen3:1.7b")
messages = [
    SystemMessage("你是一个SEO专家"),
    HumanMessage("给我科普一下SEO的基础只是")
]

result = llm.invoke(messages) #返回的是AImessage类型
print(result)