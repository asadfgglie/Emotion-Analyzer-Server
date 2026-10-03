from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated, Literal, Union, Optional

import config


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


# ---------- /v1/systemone (參考 TypeSafe jev 的 API 設計，底層全部委託給 /analyze) ----------
class _LabelQuestion(BaseModel):
    instructions: str = "這是一句會使用{}表情說出來的話。"
    """
    hypothesis template，需包含 `{}`
    """
    weights: Optional[list[float]] = None
    """
    同 /analyze 的 `weights`。
    """

    @model_validator(mode='after')
    def format(self):
        if self.instructions.find('{}') == -1:
            raise ValueError('`instructions` should contain `{}` to put criteria in it.')
        return self


class NoulQuestion(_LabelQuestion):
    """是非題：單一 NLI。`instructions` 直接作為 hypothesis（完整句子，不需要 `{}`）。"""
    type: Literal['noul']
    criteria: dict[Literal['true', 'false'], str] = Field(min_length=2, max_length=2)


class ChoiceQuestion(_LabelQuestion):
    """多個標籤選一個：等同 /analyze 的 `multi_label=False`（softmax）。"""
    type: Literal['choice']
    criteria: dict[str, Optional[str]] = Field(min_length=1)


class ScoreQuestion(_LabelQuestion):
    """等同 /analyze 的 `multi_label=True`：每個標籤獨立計分，依機率由高到低排序。"""
    type: Literal['score']
    criteria: list[str] = Field(min_length=1)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator='type')]


class SystemOneRequest(BaseModel):
    state: str
    """
    要分類的一句話。
    """
    model: str = config.MODEL_NAME
    """
    僅為相容而接受，實際一律使用 `config.MODEL_NAME`。
    """
    questions: dict[str, Question]


class SystemOneAnswer(BaseModel):
    type: Literal['noul', 'choice', 'score']
    noul: Optional[float] = None
    """
    hypothesis 成立（entailment）的機率，僅 noul。
    """
    choice: Optional[str] = None
    """
    機率最高的標籤，僅 choice / score。
    """
    probabilities: Optional[dict[str, float]] = None
    """
    各標籤機率，依由高到低排序，僅 choice / score。
    """
    confidence: float


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, SystemOneAnswer]
