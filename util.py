from langchain.messages import AIMessage

def extract_ai_response(res):
    # 逆序遍历消息列表，优先获取最后一条有效的 AIMessage
    for message in reversed(res["messages"]):
        if isinstance(message, AIMessage) and hasattr(message, "content") and message.content.strip():
            return message.content
    return None