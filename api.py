import logging
import sys
import time
import warnings

import fastapi
import numpy as np
import torch.cuda
from transformers import pipeline, Pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing_extensions import Optional, Union

import config
from schema import AnalyzeRequest, AnalyzeResponse, AnalyzeTestResponse

if config.USE_TRANSLATOR:
    from googletrans import Translator
    translator = Translator()

app = fastapi.FastAPI()
pipe: Optional[Pipeline] = None
warnings.filterwarnings(action='ignore', category=UserWarning, message='Length of IterableDataset')


@app.post('/analyze', response_model=Union[AnalyzeTestResponse, list[AnalyzeResponse], AnalyzeResponse])
async def analyze(request: AnalyzeRequest):
    logging.info('start analyze...')
    old_seq = request.sequences

    t1 = time.time()
    if config.USE_TRANSLATOR:
        request.sequences = translator.translate(request.sequences).text if isinstance(request.sequences, str) else [trans.text for trans in translator.translate(request.sequences)]
        t2 = time.time()
        logging.info(f'translate text: {old_seq} -> {request.sequences}')
        logging.info(f'translate time: {t2 - t1}')
    else:
        t2 = t1
    response = pipe(**request.model_dump())
    t3 = time.time()

    def rerank_by_weight(data):
        if request.weights is None:
            return
        tmp = {
            k:v for k, v in zip(request.candidate_labels, request.weights)
        }

        weights = np.array([tmp[l] for l in data['labels']])
        weights_sum = weights.sum()
        weights_coefficient = weights / weights_sum

        data['scores'] = np.array(data['scores']) * weights_coefficient
        if not request.multi_label:
            data['scores'] /= data['scores'].sum()
        else:
            data['scores'] *= weights_sum
        print(data['scores'].sum())
        ls = sorted(zip(data['labels'], data['scores']), key=lambda x: x[-1], reverse=True)
        data['labels'] = []
        data['scores'] = []
        for l, s in ls:
            data['labels'].append(l)
            data['scores'].append(s)

    if isinstance(old_seq, list):
        for i, result in enumerate(response):
            result['sequence'] = old_seq[i]

            rerank_by_weight(result)
    else:
        response['sequence'] = old_seq

        rerank_by_weight(response)

    logging.info(f'inference time: {t3 - t2}')
    logging.info(f'total time: {t3 - t1}')
    logging.info(response)

    if request.return_testing_data:
        return AnalyzeTestResponse(response=response, inference_time=t3 - t2,
                                   translate_time=t2 - t1, total_time=t3 - t1,
                                   use_translator=config.USE_TRANSLATOR,
                                   use_torch_compiler=config.USE_TORCH_COMPILE,
                                   dtype_model=str(config.MODEL_DTYPE),
                                   name_model=config.MODEL_NAME)
    else:
        return response


if __name__ == '__main__':
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', force=True, handlers=[logging.StreamHandler(stream=sys.stdout)])
    logging.info('loading model...')

    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, torch_dtype=config.MODEL_DTYPE)
    if torch.cuda.is_available():
        model = model.to('cuda')
        if config.USE_TORCH_COMPILE:
            model = torch.compile(model, backend="cudagraphs")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    pipe = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer,
                    device='cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('warm up model...')
    pipe(**dict(sequences=["你是在跟我開玩笑嗎?你這該死的傢伙，我要宰了你!"],
         candidate_labels=["疑惑"], hypothesis_template="這是一句會使用{}表情說出來的話。"))

    uvicorn.run(app, host='0.0.0.0', port=20823)
