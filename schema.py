from pydantic import BaseModel
from typing_extensions import Union


class AnalyzeRequest(BaseModel):
    sequences: Union[str, list[str]]
    candidate_labels: Union[str, list[str]]
    hypothesis_template: str = "This example is {}."
    multi_label: bool = False


class AnalyzeResponse(BaseModel):
    sequence: str
    labels: list[str]
    scores: list[float]
