import logging
import time
import warnings

import fastapi
from torch import bfloat16
from transformers import pipeline, Pipeline
from typing_extensions import Optional, Union

from schema import AnalyzeRequest, AnalyzeResponse

app = fastapi.FastAPI()
pipe: Optional[Pipeline] = None
warnings.filterwarnings(action='ignore', category=UserWarning, message='Length of IterableDataset')


@app.post('/analyze', response_model=Union[list[AnalyzeResponse], AnalyzeResponse])
async def analyze(request: AnalyzeRequest):
    logging.info('start analyze...')
    t1 = time.time()
    r = pipe(**request.model_dump())
    logging.info(f'time cost: {time.time() - t1}')
    return r


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
