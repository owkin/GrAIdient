import cv2
import pickle
import argparse
import numpy as np
from typing import List


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
            ret_images[label_tmp] += R.flatten().tolist() + G.flatten().tolist() + B.flatten().tolist()
        else:
            ret_images[label_tmp] += data.tolist()

    return ret_images[label]


def load_CIFAR_data(
    folder: str,
    data_file: int,
    label: int,
    size: int
) -> List[int]:
    with open(f"{folder}/data_batch_{data_file}", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return extract_images(data_dict=dict, label=label, size=size)


def load_CIFAR_test(
    folder: str,
    label: int,
    size: int
) -> List[int]:
    with open(f"{folder}/test_batch", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return extract_images(data_dict=dict, label=label, size=size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='input data directory', default='')
    args = parser.parse_args()

    _ = load_CIFAR_data(
        folder=args.dir,
        data_file=1,
        label=0,
        size=32
    )
    _ = load_CIFAR_test(
        folder=args.dir,
        label=0,
        size=32
    )
