from pydantic import BaseModel, Field, model_validator
from typing_extensions import Union, Optional


class AnalyzeRequest(BaseModel):
    sequences: Union[str, list[str]]
    """
    A list of sequences or single str. Since Pipline mechanism, inference time will increase when add more and more sequences.
    Big O is O(len(candidate_labels)), if sequences is list of str it will become O(len(candidate_labels) * len(sequences))
    """
    candidate_labels: list[str]
    """
    A list of candidate labels. Since NLI mechanism, inference time will increase when add more and more candidate labels.
    Big O is O(len(candidate_labels)), if sequences is list of str it will become O(len(candidate_labels) * len(sequences))
    """
    hypothesis_template: str = "這是一句會使用{}表情說出來的話。"
    """
    The hypothesis use in NLI. It should contain '{}' to put label in it.
    """
    multi_label: bool = False
    """
    If it is False, use softmax for each label's entailment score to get normalize score. 
    Otherwise return original entailment score.
    """

    return_testing_data: bool = False
    """
    Set this `Ture` to get inference time, total_time, translate_time, dtype_model, etc.
    """
    weights: Optional[list[float]] = Field(None, exclude=True, examples=[[1]],
                                           description="""
Set this as None will not change model output. 
If your model has some bias at specific label, you can use this to balance result.
All weight should great than zero and len(weights) must equal to len(candidate_labels).
""")

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