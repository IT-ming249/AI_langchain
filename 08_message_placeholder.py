from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from constant import DEEPSEEK_API_KEY
llm = ChatDeepSeek(
    model="deepseek-chat",  # 或者 "deepseek-reasoner"
    api_key=DEEPSEEK_API_KEY,
    temperature=0.2,  # 新增：控制随机性
    max_tokens=2048,  # 新增：控制最大生成token数
    # timeout=30,  # 可选：超时设置
)

prompt_template = ChatPromptTemplate.from_messages(
    messages= [
        ("system", "你是一个SEO专家"),
        MessagesPlaceholder(variable_name="added_message"), #这是一个占位符，放没想好的提示词，或者附加需求
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
result = llm.invoke(prompt)
print(result.content)
