# Emotion Analyzer Server

**Zero-shot Text Classifier RESTFUL Server**

---

At first time start server, use `setup.bat` to install requirements.

After install requirements, use `start.bat` to start the server.

All server's endpoint can check at `http://127.0.0.1:20823/docs`.

If you want to rerun the speed benchmark, please use `speed_benchmark.py` to test it after starting the server.


## Endpoints

### `POST /analyze`

The original zero-shot classification endpoint. Kept for backward compatibility.

### `POST /v1/systemone`

An endpoint whose request/response design follows [TypeSafe's `systemone` API](https://docs.typesafe.ai/api).
It does **not** call TypeSafe; every question is converted into one `/analyze` request on the local model.
`model` is accepted only for compatibility, the server always uses `config.MODEL_NAME`.

`state` is the sentence to classify. `questions` is a map of question id to question:

| `type` | Equivalent `/analyze` request | Response |
|--------|-------------------------------|----------|
| `choice` | `multi_label=False` (softmax over the criteria). `criteria` is `{label: description or null}`. | `choice`, `probabilities` (sorted desc), `confidence` |
| `score` | `multi_label=True` (independent score per label). `criteria` is a list of labels. | `choice`, `probabilities` (sorted desc), `confidence` |
| `noul` | `multi_label=True` with two labels, one per `criteria["true"]` / `criteria["false"]`. | `noul` (score of the `true` label), `confidence` (larger of the two labels' scores divided by their sum) |

For `choice` and `score`, `instructions` is the hypothesis template and must contain `{}` (default: `這是一句會使用{}表情說出來的話。`).
`weights` is optional and works like `/analyze`'s `weights`.
For `noul`, `instructions` is also a template: it is formatted with `criteria["true"]` and `criteria["false"]` to get the two hypotheses.
For `choice`, a label with a description is sent to the model (and returned in `choice` / `probabilities`) as `label:description`.

```json
{
  "state": "你是在跟我開玩笑嗎?",
  "questions": {
    "emotion": {"type": "choice", "criteria": {"生氣": null, "疑惑": null}},
    "ranking": {"type": "score", "criteria": ["生氣", "疑惑", "開心"]},
    "angry": {"type": "noul", "instructions": "說話的人{}。", "criteria": {"true": "很生氣", "false": "不生氣"}}
  }
}
```
