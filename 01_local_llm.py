from langchain_ollama import ChatOllama

#调用本地大模型，需要ollama部署本地大模型以后使用

llm = ChatOllama(model="qwen3:1.7b")
message = llm.invoke("我是一个开发牛马，说点让我开心的事情")
print(message)
