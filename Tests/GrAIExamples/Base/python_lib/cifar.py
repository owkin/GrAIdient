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
    """
    Load and flatten data from CIFAR data.

    Parameters
    ----------
    data_dict:
        The dictionary containing CIFAR data.
    label: int
        The label we want the data associated to.
    size: int
        The size of the images.

    Returns
    -------
    List[int]
        The list of flatten images with inner shape:
        (batch, channel, height, width).
    """
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


def load_CIFAR_data(
    data_file: int,
    label: int,
    size: int
) -> List[int]:
    """
    Load and flatten data from CIFAR train dataset.

    Parameters
    ----------
    data_file:
        The data file name containing part of the CIFAR training data.
    label: int
        The label we want the data associated to.
    size: int
        The size of the images.

    Returns
    -------
    List[int]
        The list of flatten images with inner shape:
        (batch, channel, height, width).
    """
    data_dir = Path(__file__).parent.parent.parent.parent.resolve() / \
        "data" / "in" / "cifar-10-batches-py"

    with open(f"{data_dir}/data_batch_{data_file}", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return extract_images(data_dict=dict, label=label, size=size)


def load_CIFAR_test(
    label: int,
    size: int
) -> List[int]:
    """
    Load and flatten data from CIFAR test dataset.

    Parameters
    ----------
    label: int
        The label we want the data associated to.
    size: int
        The size of the images.

    Returns
    -------
    List[int]
        The list of flatten images with inner shape:
        (batch, channel, height, width).
    """
    data_dir = Path(__file__).parent.parent.parent.parent.resolve() / \
        "data" / "in" / "cifar-10-batches-py"

    with open(f"{data_dir}/test_batch", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return extract_images(data_dict=dict, label=label, size=size)
