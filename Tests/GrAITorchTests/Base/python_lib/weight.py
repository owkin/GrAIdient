import torch
import numpy as np
from typing import List, Tuple

from python_lib.model import ModelTest1, ModelTest2


def _flatten_weights(
    layer_weights: np.ndarray
) -> Tuple[List[float], List[int]]:
    """
    Flatten weights and biases.

    Parameters
    ----------
    layer_weights: np.ndarray
        The weights to flatten.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    weights = layer_weights.data.cpu().numpy()

    weights_list = weights.flatten().tolist()
    dims_list = list(weights.shape)

    return weights_list, dims_list


def _extract_weights(
    model: torch.nn.Module
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases.

    Parameters
    ----------
    model: torch.nn.Module
        The module to get the weights and biases from.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    model_weights = model.state_dict()

    layers_weights: List[List[float]] = []
    layers_dims: List[List[int]] = []
    for name, layer_weights in model_weights.items():
        print(f"Extracting weigths {name}.")
        weights_list, dims_list = _flatten_weights(layer_weights)

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)

    return layers_weights, layers_dims


def load_test1_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest1.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest1()
    return _extract_weights(model)


def load_test2_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest2.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest2()
    return _extract_weights(model)
