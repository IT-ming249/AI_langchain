from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import get_usage_metadata_callback
from constant import DASHSCOPE_API_KEY

api_key = DASHSCOPE_API_KEY
# 模型接口
# 01 init_chat_model
model1 = init_chat_model(
    model_provider="openai", # 模型供应商
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    model="qwen3-max"
)

# 02 ChatOpenAI, 这种方法不用传模型供应商
model2 = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    model="qwen-plus"
)

# invoke 一次性输出
def invoke_demo():
    res = model1.invoke("用一句话解释 LangChain 的价值。")
    print(res.content)
    res = model2.invoke("用一句话解释 LangChain 的价值。")

# stream 流式输出
def stream_demo():
    for chunk in model1.stream("用2句话解释 LangChain 的价值。"):
        print(chunk.content, end='', flush=True)  # end ,flush是print函数的参数

# batch批量请求, 多段提示词多次请求
def batch_demo():
    prompts = ["写 1 个标题", "写 1 个副标题", "写 1 个课程卖点"]
    results = model1.batch(prompts)
    for r in results:
        print("-", r.content)

# token消耗统计
def token_usage_demo():
    # 1. 上下文管理器
    with get_usage_metadata_callback() as cb:
        res = model1.invoke("Hello how are you?")
        print(res.content)
        res2 = model2.invoke("Hello how are you?")
        print(res2.content)
        print(cb.usage_metadata)



if __name__ == "__main__":
    token_usage_demo()
