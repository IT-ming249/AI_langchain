from langchain_ollama import ChatOllama

#调用本地大模型，需要ollama部署本地大模型以后使用

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.7,  # 新增参数，建议设置
    num_predict=512,  # 可选：控制生成的最大token数
)
message = llm.invoke("我是一个开发牛马，说点让我开心的事情")
print(message.content)
