import torch
import numpy as np
from typing import List, Tuple

from python_lib.model import ModelTest1


def _flatten_weights(layer_weights: np.ndarray) -> Tuple[List[float], List[int]]:
    weights = layer_weights.data.cpu().numpy()

    weights_list = weights.flatten().tolist()
    dims_list = list(weights.shape)

    return weights_list, dims_list


def _extract_weights(model: torch.nn.Module) -> Tuple[List[List[float]], List[List[int]]]:
    model_weights = model.state_dict()

    layers_weights: List[List[float]] = []
    layers_dims: List[List[int]] = []
    for name, layer_weights in model_weights.items():
        print(f"Extracting weigths {name}.")
        weights_list, dims_list = _flatten_weights(layer_weights)

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)
        """
        if len(dims_list) == 1:
            for i in range(dims_list[0]):
                assert weights_list[i] == weights[i]

        elif len(dims_list) == 4:
            for i in range(dims_list[0]):
                for j in range(dims_list[1]):
                    for k in range(dims_list[2]):
                        for l in range(dims_list[3]):
                            assert weights_list[l +
                                                k * dims_list[3] +
                                                j * dims_list[2] * dims_list[3] +
                                                i * dims_list[1] * dims_list[2] * dims_list[3]
                                                ] == weights[i, j, k, l]
        else:
            raise RuntimeError
        """

    return layers_weights, layers_dims


def load_test1_weights() -> Tuple[List[List[float]], List[List[int]]]:
    torch.manual_seed(42)
    model = ModelTest1()
    return _extract_weights(model)
