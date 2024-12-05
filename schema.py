from pydantic import BaseModel
from typing_extensions import Union, Optional


class AnalyzeRequest(BaseModel):
    sequences: Union[str, list[str]]
    candidate_labels: Union[str, list[str]]
    hypothesis_template: str = "這是一句會使用{}表情說出來的話。"
    multi_label: bool = False

    return_testing_data: bool = False


class AnalyzeResponse(BaseModel):
    sequence: str
    labels: list[str]
    scores: list[float]


class AnalyzeTestResponse(BaseModel):
    response: Union[AnalyzeResponse, list[AnalyzeResponse]]
    inference_time: float
    translate_time: float
    total_time: float
