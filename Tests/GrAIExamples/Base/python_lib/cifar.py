import cv2
import pickle
import torch
import torchvision
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose
)


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


def load_CIFAR_train(
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


class MaskSampler(torch.utils.data.sampler.Sampler):
    """
    Sampler of indices that is based on a mask.

    Parameters
    ----------
    mask: np.ndarray
        Base mask of the indices to consider.
    """

    def __init__(self, mask: np.ndarray):
        self.indices = np.nonzero(mask)[0]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def iter_CIFAR(
    train: bool,
    batch_size: int,
    label: int,
    shuffle: bool
):
    """
    Build an iterator on CIFAR dataset.

    Parameters
    ----------
    train: bool
        Train or test dataset.
    batch_size: int
        The batch size.
    label: int
        The label we want the data associated to.
    shuffle: bool
        Whether to shuffle indices of data.

    Returns
    -------
    An iterator on CIFAR dataset.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = Compose([
        ToTensor(),
        Normalize(mean, std)
    ])
    data_dir = Path(__file__).parent.parent.parent.parent.resolve() / \
        "data" / "in"
    cifar = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    indices = np.array(cifar.targets) == label

    return iter(torch.utils.data.DataLoader(
        cifar, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        sampler=MaskSampler(indices)
    ))


def next_tensor_CIFAR(iterator) -> Optional[torch.Tensor]:
    """
    Load next data from a CIFAR iterator.

    Parameters
    ----------
    iterator
        The CIFAR dataset iterator.

    Returns
    -------
    torch.Tensor
        The images tensor with inner shape:
        (batch, channel, height, width).
    """
    try:
        samples, _ = next(iterator)
    except StopIteration:
        return None
    return samples


def next_data_CIFAR(iterator) -> Tuple[List[float], int]:
    """
    Load and flatten next data from a CIFAR iterator.

    Parameters
    ----------
    iterator
        The CIFAR dataset iterator.

    Returns
    -------
    List[int]
        The list of flatten images with inner shape:
        (batch, channel, height, width).
    int
        The batch size of data.
    """
    samples = next_tensor_CIFAR(iterator)
    if samples is not None:
        return samples.flatten().tolist(), len(samples)
    else:
        return [], 0
