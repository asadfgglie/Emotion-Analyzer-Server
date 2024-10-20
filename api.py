import logging
import time
import warnings

import fastapi
from torch import bfloat16
from transformers import pipeline, Pipeline
from typing_extensions import Optional, Union

from schema import AnalyzeRequest, AnalyzeResponse
from googletrans import Translator

translator = Translator()
app = fastapi.FastAPI()
pipe: Optional[Pipeline] = None
warnings.filterwarnings(action='ignore', category=UserWarning, message='Length of IterableDataset')


@app.post('/analyze', response_model=Union[list[AnalyzeResponse], AnalyzeResponse])
async def analyze(request: AnalyzeRequest):
    logging.info('start analyze...')
    t1 = time.time()
    old_seq = request.sequences
    request.sequences = translator.translate(request.sequences).text if isinstance(request.sequences, str) else [trans.text for trans in translator.translate(request.sequences)]
    logging.info(f'translate text: {old_seq} -> {request.sequences}')
    logging.info(f'translate time: {time.time() - t1}')
    response = pipe(**request.model_dump())
    if isinstance(old_seq, list):
        for i, result in enumerate(response):
            result['sequence'] = old_seq[i]
    else:
        response['sequence'] = old_seq
    logging.info(f'total time cost: {time.time() - t1}')
    return response


if __name__ == '__main__':
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logging.info('loading model...')
    pipe = pipeline("zero-shot-classification", model="VivekMalipatel23/mDeBERTa-v3-base-text-emotion-classification",
                    model_kwargs={
                        'load_in_4bit': True
                    }, torch_dtype=bfloat16)
    uvicorn.run(app, host='0.0.0.0', port=20823)
