from pydantic import BaseModel, Field, model_validator
from typing_extensions import Union, Optional


class AnalyzeRequest(BaseModel):
    sequences: Union[str, list[str]]
    candidate_labels: list[str]
    hypothesis_template: str = "這是一句會使用{}表情說出來的話。"
    multi_label: bool = False

    return_testing_data: bool = False
    weights: Optional[list[float]] = Field(None, exclude=True)

    @model_validator(mode='after')
    def format(self):
        if self.weights is not None:
            if len(self.weights) != len(self.candidate_labels):
                raise ValueError('`weights` length should same as candidate_labels number, or weights=None')
            else:
                for i in self.weights:
                    if i <= 0:
                        raise ValueError('weight should great than zero!')

        return self


class AnalyzeResponse(BaseModel):
    sequence: str
    labels: list[str]
    scores: list[float]


class AnalyzeTestResponse(BaseModel):
    response: Union[AnalyzeResponse, list[AnalyzeResponse]]
    inference_time: float
    translate_time: float
    total_time: float
    use_translator: bool
    use_torch_compiler: bool
    name_model: str
    dtype_model: str