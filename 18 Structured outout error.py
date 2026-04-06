from typing import Union

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError, \
    MultipleStructuredOutputsError
from llm import model


class ContactInfo(BaseModel):
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")


class EventDetails(BaseModel):
    event_name: str = Field(description="事件名称")
    date: str = Field(description="事件日期")

class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "格式出了问题。再试一次。"
    elif isinstance(error, MultipleStructuredOutputsError):
        return "返回了多个结构化输出。选一个最相关的。"
    else:
        return f"错误: {str(error)}"


def multiple_output_errorr():
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(
            Union[ContactInfo, EventDetails],
            handle_errors=custom_error_handler # 这是自定义的结构化错误处理方式
        ),  # 默认 handle_errors=True
    )

    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "提取信息: 张三(zhangsan@qq.com)将于3月15日举办技术大会。",
        }]
    })
    messages = response["messages"]
    for message in messages:
        message.pretty_print()

def schema_validate_error():
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(ProductRating),  # 默认 handle_errors=True
        system_prompt=(
            "您是解析产品评论的得力助手。 "
            "不要虚构任何字段或数值。"
        ),

    )

    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "解析一下：很棒的产品，10/10！",
        }]
    })
    messages = response["messages"]
    for message in messages:
        message.pretty_print()

if __name__ == '__main__':
    # multiple_output_errorr()
    schema_validate_error()
