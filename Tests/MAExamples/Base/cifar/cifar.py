import cv2
import pickle
import numpy as np
from typing import List
from pathlib import Path


def extract_images(
    data_dict,
    label: int,
    size: int
) -> List[int]:
    ret_images: List[List[int]] = [[] for _ in range(10)]

    for label_tmp, data_tmp in zip(data_dict[b"labels"], data_dict[b"data"]):
        data = data_tmp
        if size != 32:
            data = cv2.merge([
                np.reshape(data[0: 32 * 32], (32, 32)),
                np.reshape(data[32 * 32: 2 * 32 * 32], (32, 32)),
                np.reshape(data[2 * 32 * 32:], (32, 32))
            ])
            data = cv2.resize(data, (size, size))
            R, G, B = cv2.split(data)
            ret_images[label_tmp] += \
                R.flatten().tolist() + \
                G.flatten().tolist() + \
                B.flatten().tolist()
        else:
            ret_images[label_tmp] += data.tolist()

    return ret_images[label]


def test_dir() -> str:
    return Path(__file__).as_posix()


def load_CIFAR_data(
    data_file: int,
    label: int,
    size: int
) -> List[int]:
    data_dir = Path(__file__).parent.parent.resolve() / "data"

    with open(f"{data_dir}/data_batch_{data_file}", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return extract_images(data_dict=dict, label=label, size=size)


def load_CIFAR_test(
    label: int,
    size: int
) -> List[int]:
    data_dir = Path(__file__).parent.parent.resolve() / "data"

    with open(f"{data_dir}/test_batch", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return extract_images(data_dict=dict, label=label, size=size)
