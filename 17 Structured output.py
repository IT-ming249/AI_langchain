from typing import Literal

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from llm import model
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy, StructuredOutputValidationError, \
    MultipleStructuredOutputsError


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(
        description="The rating of the product", ge=1, le=5
    )
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )
    key_points: list[str] = Field(
        description="The key points of the review. Lowercase, 1-3 words each."
    )


def auto_struct_output_demo():
    agent = create_agent(
        model=model,
        response_format=ProviderStrategy(schema=ContactInfo)  # 自动选择结构化策略，模型能力强的可以直接这样用
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "从以下内容中提取信息：张三，zhangsan@qq.com，18899990000"
        }]
    })
    print(result["structured_response"].name)

def tool_struct_output_demo():
    agent = create_agent(
        model=model,
        tools=[],  # 可以同时使用其他业务工具
        response_format=ToolStrategy(ProductReview), # 模型能力不强时使用，出错概率较大
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive"
        }]
    })
    print(result["structured_response"])

if __name__ == '__main__':
    auto_struct_output_demo()
    # tool_struct_output_demo()