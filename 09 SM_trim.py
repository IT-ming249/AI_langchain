from dataclasses import dataclass

import tiktoken
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from llm import model
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage
from util import extract_ai_response

def _count_message_tokens(message: BaseMessage, enc: tiktoken.Encoding) -> int:
    """粗略 token 计数：实战中用“近似估算 + 安全余量”通常足够。"""
    # BaseMessage 结构随版本可能略有不同，这里用最稳妥的方式转字符串
    content = getattr(message, "content", "")
    return len(enc.encode(str(content)))


@dataclass
class TrimContext:
    token_budget = 100


#before_model是调用大模型之前的拦截器
@before_model
def trim_messages(state: AgentState, runtime: Runtime[TrimContext]):
    # 有state才能获取大模型的所有message
    # 这里面的runtime不是Toolruntime, 调了工具才用Toolruntime

    ## 1. 获取所有message
    messages = state.get("messages")

    ## 2. 提取第一条消息，因为这条消息通常是系统提示词，不需要裁剪掉
    first_message = messages[0]
    rest_message = messages[1:]

    ## 3.统计第一条消息的token
    # 获取分词器,下面是所有可选择的编码方式，每个模型的编码方式可能不一样，所以这里只是粗略统计token
    # ['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base', 'o200k_harmony']
    enc = tiktoken.get_encoding("cl100k_base")
    # tiktoken里面自带算法计算token用量
    total = _count_message_tokens(first_message, enc)

    kept_messages = []
    ## 4.从最后一条消息开始统计token消耗,因为既然要裁剪，肯定保留最新的消息比较好
    for message in reversed(rest_message):
        usage = _count_message_tokens(message, enc)
        if total + usage > runtime.context.token_budget:
            break
        kept_messages.append(message)
        total = total + usage

    ## 5.将要保留的消息进行反转，因为刚刚反转过了，现在要赚回来
    kept = [first_message] + list(reversed(kept_messages))

    ## 6.没发生裁剪则不更新 state（减少无意义的 checkpoint 写入）
    if len(kept) == len(messages):
        return None

    ## 7.干掉所有消息，再把要保留的拼接到消息末尾
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *kept,
        ]
    }

if __name__ == '__main__':
    agent = create_agent(
        model=model,
        context_schema=TrimContext,
        middleware=[trim_messages],
        checkpointer=InMemorySaver()
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": [
        HumanMessage(content="你好，我的名字是张三")
    ]}, config=config, context=TrimContext())

    agent.invoke({"messages": [
        HumanMessage("请写一首关于小猫的打油诗")
    ]}, config=config, context=TrimContext())

    agent.invoke({"messages": [
        HumanMessage("也给小狗写一首打油诗")
    ]}, config=config, context=TrimContext())

    final_response = agent.invoke({"messages": [
        HumanMessage('我的名字是什么？')
    ]}, config=config, context=TrimContext())

    print(extract_ai_response(final_response))



