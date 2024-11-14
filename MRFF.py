import numpy as np
import pandas as pd
import math
import random
import glob
import os

n = 3  # redundance
l = 8  # bit length
T = 400  # time interval(ms)

is_extracted = []
ber = []


# Caculate the bit error rate
def get_ber(message, extraction):
    message, extraction = np.array(message), np.array(extraction)
    error = (message != extraction)
    ber = np.sum(error) / len(message)
    return ber


def extract(timestamp):
    packets_per_slot = [0] * (l * n * 3)
    for t in timestamp:
        slot_num = int(t // T)
        if slot_num >= len(packets_per_slot):
            break
        packets_per_slot[slot_num] += 1

    bits = []
    for i in range(0, len(packets_per_slot), 3):
        if packets_per_slot[i] >= packets_per_slot[i + 1]:
            bits.append(1)
        else:
            bits.append(0)

    extraction = []
    for i in range(0, len(bits), 3):
        ones = bits[i:i + 3].count(1)
        if ones >= 2:
            extraction.append(1)
        else:
            extraction.append(0)

    return extraction


# Caculate which bit and which slot the timestamp belongs to
def get_bit_and_slot(t):
    slot_num = t // T
    bit_num = slot_num // (3 * n)
    return int(bit_num), int(slot_num % 3)


def experiment_once(timestamp):
    message = [random.randint(0, 1) for _ in range(l)]

    # embed
    for i in range(len(timestamp)):
        bit_num, slot = get_bit_and_slot(timestamp[i])
        if bit_num >= len(message):
            break
        if slot == 0 and message[bit_num] == 0:
            timestamp[i] += T
        elif slot == 1 and message[bit_num] == 1:
            timestamp[i] += T

    # add noise
    noise = np.random.laplace(10, 3, (len(timestamp, )))
    timestamp += noise

    # extract
    extraction = extract(timestamp)

    is_extracted.append(message == extraction)
    ber.append(get_ber(message, extraction))


total_time = T * 3 * n * l
data_path = './log2'

files = glob.glob(os.path.join(data_path, '*.csv'))
for file in files:
    data_df = pd.read_csv(file, header=None)
    timestamp = data_df[0].values
    timestamp = (timestamp - timestamp[0]) * 1000
    if timestamp[-1] < total_time:
        continue

    experiment_once(timestamp)

print("ER:", is_extracted.count(True) / len(is_extracted))
print("BER:", sum(ber) / len(ber))
