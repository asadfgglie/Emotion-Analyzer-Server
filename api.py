import logging
import sys
import time
import warnings

import fastapi
import torch.cuda
from torch import float16
from transformers import pipeline, Pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing_extensions import Optional, Union

from schema import AnalyzeRequest, AnalyzeResponse, AnalyzeTestResponse
import config

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

    if isinstance(old_seq, list):
        for i, result in enumerate(response):
            result['sequence'] = old_seq[i]
    else:
        response['sequence'] = old_seq

    logging.info(f'inference time: {t3 - t2}')
    logging.info(f'total time: {t3 - t1}')

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
         candidate_labels=["疑惑", "生氣", "害羞", "無語", "自然"],
         multi_label=True, hypothesis_template="這是一句會使用{}表情說出來的話。"))

    uvicorn.run(app, host='0.0.0.0', port=20823)
