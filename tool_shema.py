from typing import Literal
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """查询天气的入参"""
    location: str = Field(description="城市名或经纬度")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius",description="温度单位",)
    include_forecast: bool = Field(default=False,description="是否包含 5 天预报",)