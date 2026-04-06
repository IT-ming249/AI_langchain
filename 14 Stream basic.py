from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import model
from langchain_core.messages import AIMessageChunk
import asyncio


"""
chunk1 = AIMessageChunk(content="Hello")
chunk2 = AIMessageChunk(content=" World")
full_message = chunk1 + chunk2  # AIMessageChunk(content="Hello World")
"""


def sync_chat():
    for chunck in model.stream("请写一首关于夏天的短诗"):
        print(chunck.content, end="", flush=True)


# 使用 astream 进行异步流式处理
async def stream_chat():
    async for chunk in model.astream("请写一首关于春天的短诗"):
        # chunk 是 AIMessageChunk 对象
        print(chunk.content, end="", flush=True)


async def accumulate_chunks():
    full_message = None
    async for chunk in model.astream("解释什么是量子计算"):
        if full_message is None:
            full_message = chunk
        else:
            full_message += chunk

    # 最终得到完整的消息
    print(full_message.content)

async def stream_chain():
    prompt = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话")
    parser = StrOutputParser()

    # 构成 LCEL 链：Prompt -> Model -> Parser
    # 类中含stream方法的都可以用流式输出
    chain = prompt | model | parser

    async for chunk in chain.astream({"topic": "程序员"}):
        # 由于使用了 StrOutputParser，这里的 chunk 是解析后的字符串
        print(chunk, end="", flush=True)



if __name__ == '__main__':
    #sync_chat() #同步
    #asyncio.run(stream_chat()) #异步
    #asyncio.run(accumulate_chunks())
    asyncio.run(stream_chain()) #LECL