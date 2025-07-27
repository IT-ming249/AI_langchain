from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

## 提示词模板 (message)

# 1. 元组
messages1 = [
    ("system", "你是一个{field}领域的专家"),
    ("user", "给我科普一下{topic}知识")
]

#chat_prompt = ChatPromptTemplate(messages) 上下两种都可以
chat_prompt = ChatPromptTemplate.from_messages(messages1)
prompt = chat_prompt.invoke({"field": "SEO", "topic": "谷歌搜索引擎优化"})

print(prompt)
print('--------------------------------')
# 2. message对象
# 如果模板中存在变量， 就不能使用message对象，转而使用元组或字典形式
messages2 = [
    SystemMessage("你是一个{field}领域的专家"),
    HumanMessage("给我科普一下{topic}知识")
]

chat_prompt2 = ChatPromptTemplate.from_messages(messages2)
prompt2 = chat_prompt2.invoke({"field": "SEO", "topic": "谷歌搜索引擎优化"})
print(prompt2)


