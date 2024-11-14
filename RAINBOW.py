import os
import glob
import pandas as pd
import numpy as np
import random

a = 9
length = 1200
threshold = 0.62


# Calculate normalized correlation
def normalization(a, b):
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return dot / (norm_a * norm_b)


# Calculate true/false positive rate
def rate(a):
    return a.count(True) / len(a)


def experiment_once(ipd):
    global res_without_w, res_with_w
    w = [random.choice([a, -a])
         for _ in range(length)]  # Generate random delay
    delta = np.random.laplace(10, 3, length)  # Generate random jitter

    received_ipd = ipd + delta  # without watermarking
    watermarked_ipd = ipd + w + delta  # with watermarking

    y1 = received_ipd - ipd
    y2 = watermarked_ipd - ipd

    normal_y1 = normalization(y1, w)  # without watermarking
    normal_y2 = normalization(y2, w)  # with watermarking

    res_without_w.append(normal_y1 > threshold)
    res_with_w.append(normal_y2 > threshold)


data_path = './log2'

res_without_w, res_with_w = [], []
files = glob.glob(os.path.join(data_path, '*.csv'))
for file in files:
    data_df = pd.read_csv(file, header=None)
    timestamp = data_df[0].values
    ipd = np.diff(timestamp)[:length]
    experiment_once(ipd)

print(f'TPR: {rate(res_with_w)}')
print(f'FPR: {rate(res_without_w)}')
