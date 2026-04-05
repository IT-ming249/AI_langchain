from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from llm import model


"""
SummarizationMiddleware的trigger和keep参数介绍：
trigger：触发条件。
("messages", 50)：当消息数量达到50时触发。
("tokens", 3000)：当token数量达到3000时触发。
("fraction", 0.8)：当token数量达到大模型最大输入token的0.8时触发。
keep：保留的消息数量。
("messages", 20)：保持最近20条消息，不会被总结。
("tokens", 3000)：保持最近3000个token不会被总结。
("fraction", 0.3)：保持当前模型最大输入token的30%不会被总结。

"""

def main():
    agent = create_agent(
        model=model,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 100),
                keep=("messages", 2),
            )
        ],
        checkpointer=InMemorySaver()
    )
    config = RunnableConfig(configurable={"thread_id": "1"})

    agent.invoke({"messages": "你好，我的名字是张三"}, config)
    agent.invoke({"messages": "请写一首关于小猫的打油诗"}, config)
    agent.invoke({"messages": "请写一首关于小狗的打油诗"}, config)
    final_response = agent.invoke({"messages": "我的名字是什么？"}, config)

    print(final_response["messages"])

if __name__ == '__main__':
    main()
