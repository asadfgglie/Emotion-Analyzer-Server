import time
from tqdm import tqdm
import numpy as np
import requests
from datasets import load_dataset

from schema import AnalyzeRequest, AnalyzeTestResponse

url = 'http://26.171.83.203:20823/analyze'
dataset_name = 'asadfgglie/BanBan_2024-10-17-facial_expressions'
session = requests.session()

# warm up connection
session.post(url, json=AnalyzeRequest(sequences='hello, world!',
                                      candidate_labels=['興奮', '難過', '開心', '自然']).model_dump())


class Timer:
    def __init__(self):
        self.inference_time: list[float] = []
        self.translate_time: list[float] = []
        self.total_time: list[float] = []
        self.response_time: list[float] = []
        self.label_number: list[int] = []

    def add_data(self, response: AnalyzeTestResponse, response_time: float, label_num: int):
        self.inference_time.append(response.inference_time)
        self.total_time.append(response.total_time)
        self.translate_time.append(response.translate_time)
        self.response_time.append(response_time)
        self.label_number.append(label_num)

    def clear(self):
        self.inference_time.clear()
        self.total_time.clear()
        self.translate_time.clear()
        self.response_time.clear()

    @staticmethod
    def mean(data: list[float]):
        return np.mean(data)

    @staticmethod
    def std(data: list[float]):
        return np.std(data)

    @property
    def mean_inference_time(self):
        return self.mean(self.inference_time)

    @property
    def std_inference_time(self):
        return self.std(self.inference_time)

    @property
    def mean_total_time(self):
        return self.mean(self.total_time)

    @property
    def std_total_time(self):
        return self.std(self.total_time)

    @property
    def mean_translate_time(self):
        return self.mean(self.translate_time)

    @property
    def std_translate_time(self):
        return self.std(self.translate_time)

    @property
    def mean_response_time(self):
        return self.mean(self.response_time)

    @property
    def std_response_time(self):
        return self.std(self.response_time)

    @property
    def balance_inference_time(self):
        return np.array(self.inference_time) / np.array(self.label_number)

    @property
    def balance_mean_inference_time(self):
        return self.balance_inference_time.mean()

    @property
    def balance_std_inference_time(self):
        return self.balance_inference_time.std()


test_dataset = load_dataset(dataset_name)['train']
timer = Timer()
for ex in tqdm(test_dataset, total=test_dataset.shape[0]):
    ex['candidate_labels'] += ex['not_candidate_labels']
    ex.pop('not_candidate_labels')
    t1 = time.time()
    response = session.post(url, json=AnalyzeRequest(**ex, return_testing_data=True).model_dump()).json()
    timer.add_data(AnalyzeTestResponse(**response), time.time() - t1, len(ex['candidate_labels']))

print(f"""
========= Speed testing result =========
dataset name:           {dataset_name}
dataset size:           {test_dataset.shape}
translate time:         {timer.mean_translate_time:.4f} +- {timer.std_translate_time:.4f}
inference time:         {timer.mean_inference_time:.4f} +- {timer.std_inference_time:.4f}
total server time:      {timer.mean_total_time:.4f} +- {timer.std_total_time:.4f}
response time:          {timer.mean_response_time:.4f} +- {timer.std_response_time:.4f}
balance inference time: {timer.balance_mean_inference_time:.4f} +- {timer.balance_std_inference_time:.4f}
label number:           {timer.mean(timer.label_number):.4f} +- {timer.std(timer.label_number):.4f}
""")