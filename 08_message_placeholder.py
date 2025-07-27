from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import  ChatOllama
llm = ChatOllama(model="qwen3:1.7b")

prompt_template = ChatPromptTemplate(
    messages= [
        ("system", "你是一个SEO专家"),
        MessagesPlaceholder("added_message"), #这是一个占位符，放没想好的提示词，或者附加需求
        ("user", "{requirement}")
    ]
)

prompt = prompt_template.invoke(
    {
        "added_message": [
            ("system", "你的名字叫贾维斯")
        ],
        "requirement": "介绍一下你自己，并科普一下SEO"
    }
)
print(prompt, type(prompt))
# result = llm.invoke(prompt)
# print(result.content)
