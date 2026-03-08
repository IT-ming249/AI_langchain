from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.messages import HumanMessage
from llm import model
from util import extract_ai_response


DB_URI = "postgresql://postgres:Aa123456@localhost:5432/agent_chat_history"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 自动创建所需数据表
    # checkpointer.setup() # 只用运行一次

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,  # 使用数据库持久化短期记忆
    )

    config = {"configurable": {"thread_id": "prod-thread-1"}}
    agent.invoke(
        {"messages": [HumanMessage("请记住我喜欢深色主题")]},
        config,
    )
    response = agent.invoke({
        "messages": [HumanMessage("我喜欢什么颜色的主题？")]
    }, config)
    print(extract_ai_response(response))