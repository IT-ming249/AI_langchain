from langchain_core.prompts import PromptTemplate

## 提示词模板 (纯字符串)
# 1. 使用构造函数
prompt_template = PromptTemplate(
    template="讲一个关于{topic}方面的笑话",
    input_variable=["topic"] #指定提示词中的变量
)

prompt = prompt_template.format(topic="开发牛马")
print(prompt, type(prompt))

prompt2 = prompt_template.invoke(input={"topic": "开发牛马"})
print(prompt2, type(prompt2))
# 大模型的提示词是字符串，使用invoke构造的提示词还需要转成字符串
prompt2_str = prompt2.to_string()
print(prompt2_str, type(prompt2_str))
print("-----------------------------")

# 2. 使用from_template快速构建
txt = "讲一个关于{topic}方面的笑话"
prompt_template2 = PromptTemplate.from_template(txt) #整个方法不用指定input
print(prompt_template2.format(topic="开发牛马"))

# 3. 使用from_examples来构建(看文档去吧), 我觉得第二个更好用
