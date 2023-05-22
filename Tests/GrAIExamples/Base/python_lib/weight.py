import torch
import numpy as np
from typing import List, Tuple

from python_lib.model import SimpleAutoEncoder


def _flatten_weights(
        weights: np.ndarray
) -> Tuple[List[float], List[int]]:
    """
    Flatten weights and biases.

    Parameters
    ----------
    weights: np.ndarray
        The weights to flatten.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    weights_list = weights.flatten().tolist()
    dims_list = list(weights.shape)

    return weights_list, dims_list


def _extract_and_transpose_weights(
        modules: [torch.nn.Module]
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases.
    Transpose weights when they come from a
    ConvTranspose2d layer.

    Parameters
    ----------
    modules: [torch.nn.Module]
        The list of modules to get the weights and biases from.

    Returns
    -------
    (_, _): List[List[float]], List[List[int]]
        The flattened weights, their shape.
    """
    layers_weights: List[List[float]] = []
    layers_dims: List[List[int]] = []
    for module in modules:
        submodules = list(module.children())
        if len(submodules) > 0:
            (weights_list, dims_list) = _extract_and_transpose_weights(
                submodules
            )
            layers_weights += weights_list
            layers_dims += dims_list

        else:
            if hasattr(module, "weight"):
                if isinstance(module, torch.nn.ConvTranspose2d):
                    weights = np.transpose(
                        module.weight.detach().numpy(), (1, 0, 2, 3)
                    )
                    weights_list, dims_list = _flatten_weights(weights)

                else:
                    weights = module.weight.detach().numpy()
                    weights_list, dims_list = _flatten_weights(weights)

                layers_weights.append(weights_list)
                layers_dims.append(dims_list)

            if hasattr(module, "bias"):
                weights = module.bias.detach().numpy()
                weights_list, dims_list = _flatten_weights(weights)

                layers_weights.append(weights_list)
                layers_dims.append(dims_list)

    return layers_weights, layers_dims


def load_simple_auto_encoder_weights(
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for simple auto encoder model.

    Returns
    -------
    (_, _): List[List[float]], List[List[int]]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = SimpleAutoEncoder()
    return _extract_and_transpose_weights(list(model.children()))
