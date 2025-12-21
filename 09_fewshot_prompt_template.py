from constant import DEEPSEEK_API_KEY
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 创建提示词模板
example_prompt = PromptTemplate.from_template("问题：{question}\n{answer}")
# 创建示例(question与answer 要与example_prompt中一样，不能是question1)
examples = [
    {
        "question": "穆罕默德·阿里和艾伦·图灵谁活得更久？",
        "answer": """
这里需要后续问题吗？是的。
后续问题：穆罕默德·阿里去世时多大？
中间答案：穆罕默德·阿里去世时74岁。
后续问题：艾伦·图灵去世时多大？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里
        """
    },
    {
        "question": "craigslist的创始人什么时候出生？",
        "answer": """
这里需要后续问题吗？是的。
后续问题：Craigslist 的创始人是谁？
中间答案：Craigslist 由 Craig Newmark 创立。
后续问题：Craig Newmark 何时出生？
中间答案：Craig Newmark 出生于 1952 年 12 月 6 日。
所以最终答案是：1952 年 12 月 6 日
        """
    }, {
        "question": "乔治华盛顿的外祖父是谁？",
        "answer": """
这里需要后续问题吗？是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
后续问题：玛丽·鲍尔·华盛顿的父亲是谁？
中间答案：玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
所以最终答案是：约瑟夫·鲍尔。
        """
    }, {
        "question": "《大白鲨》和《皇家赌场》的导演都是来自同一个国家吗？",
        "answer": """
这里需要后续问题吗？是的。
后续问题：《大白鲨》的导演是谁？
中级答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
后续问题：史蒂文·斯皮尔伯格来自哪里？
中级答案：美国。
后续问题：《皇家赌场》的导演是谁？
中级答案：《皇家赌场》的导演是马丁·坎贝尔。
后续问题：马丁·坎贝尔来自哪里？
中级答案：新西兰。
所以最终答案是：否
        """
    }
]

## 给大模型提供事例
fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请根据以下示例回答问题：\n",  # 添加前缀
    suffix="问题: {my_question}",
    input_variables=["my_question"],
    example_separator="\n\n",  # 分隔符
    # template_format="f-string"  # 指定模板格式
)

prompt = fewshot_prompt.invoke({"my_question": "美国总统特朗普还能活多久"})
prompt_text = prompt.to_string()

llm = ChatDeepSeek(
    model="deepseek-chat",  # 或者 "deepseek-reasoner"
    api_key=DEEPSEEK_API_KEY,
    temperature=0.2,  # 新增：控制随机性
    max_tokens=2048,  # 新增：控制最大生成token数
    # timeout=30,  # 可选：超时设置
    # base_url="https://api.deepseek.com",  # 可选：自定义API端点
)


try:
    response = llm.invoke(prompt_text)
    print(f"\n模型回答:\n{response.content}")
except Exception as e:
    print(f"调用失败: {e}")
