import torch
import numpy as np
from typing import List, Tuple

from python_lib.model import (
    ModelTest1,
    ModelTest2,
    ModelTest4,
    ModelTest5,
    ModelTest6,
    ModelTest7,
    ModelTest8,
    ModelTest9,
    ModelTest10,
    ModelTest11,
    ModelTest12,
)


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


def _flatten_attention_weights(
    weights: np.ndarray,
    biases: np.ndarray,
    nb_heads: int
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Flatten weights and biases.

    Parameters
    ----------
    weights: np.ndarray
        The weights to flatten.
    biases: np.ndarray
        The biases to flatten.
    nb_heads: int
        Number of heads in the attention modules.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    nb_partial = int(len(weights) / nb_heads)
    layers_weights: List[List[float]] = []
    layers_dims: List[List[int]] = []

    for head in range(nb_heads):
        weights_tmp = weights[head * nb_partial: ((head + 1) * nb_partial)]

        weights_list = weights_tmp.flatten().tolist()
        dims_list = list(weights_tmp.shape)

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)

        weights_tmp = biases[head * nb_partial: ((head + 1) * nb_partial)]

        weights_list = weights_tmp.flatten().tolist()
        dims_list = list(weights_tmp.shape)

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)

    return layers_weights, layers_dims


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
    (_, _): List[List[float]], List[List[int]]
        The flattened weights, their shape.
    """
    model_weights = model.state_dict()

    layers_weights: List[List[float]] = []
    layers_dims: List[List[int]] = []
    for name, layer_weights in model_weights.items():
        print(f"Extracting weigths {name}.")
        weights_list, dims_list = _flatten_weights(
            layer_weights.data.cpu().numpy()
        )

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)

    return layers_weights, layers_dims


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


def _extract_attention_weights(
    model: torch.nn.Module,
    nb_heads: int
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases.

    Parameters
    ----------
    model: torch.nn.Module
        The module to get the weights and biases from.
    nb_heads: int
        Number of heads in the attention modules.

    Returns
    -------
    (_, _): List[List[float]], List[List[int]]
        The flattened weights, their shape.
    """
    model_weights = model.state_dict()

    layers_weights: List[List[float]] = []
    layers_dims: List[List[int]] = []

    cur_item = 0
    list_items = list(model_weights.items())

    while cur_item < len(list_items):
        name, layer_weights = list_items[cur_item]
        print(f"Extracting weigths {name}.")

        if "in_proj" in name:
            weights = layer_weights.data.cpu().numpy()
            nb_partial = int(len(weights) / 3)

            weights1 = weights[0:nb_partial]
            weights2 = weights[nb_partial: 2*nb_partial]
            weights3 = weights[2*nb_partial: 3*nb_partial]

            cur_item += 1
            name, layer_weights = list_items[cur_item]
            print(f"Extracting weigths {name}.")
            biases = layer_weights.data.cpu().numpy()

            biases1 = biases[0:nb_partial]
            biases2 = biases[nb_partial: 2 * nb_partial]
            biases3 = biases[2 * nb_partial: 3 * nb_partial]

            weights_list, dims_list = _flatten_attention_weights(
                weights=weights1, biases=biases1, nb_heads=nb_heads
            )
            layers_weights += weights_list
            layers_dims += dims_list

            weights_list, dims_list = _flatten_attention_weights(
                weights=weights2, biases=biases2, nb_heads=nb_heads
            )
            layers_weights += weights_list
            layers_dims += dims_list

            weights_list, dims_list = _flatten_attention_weights(
                weights=weights3, biases=biases3, nb_heads=nb_heads
            )
            layers_weights += weights_list
            layers_dims += dims_list

            cur_item += 1

        else:
            weights_list, dims_list = _flatten_weights(
                layer_weights.data.cpu().numpy()
            )

            layers_weights.append(weights_list)
            layers_dims.append(dims_list)

            cur_item += 1

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


def load_test4_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest4.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest4()
    return _extract_and_transpose_weights(list(model.children()))


def load_test5_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest5.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest5()
    return _extract_and_transpose_weights(list(model.children()))


def load_test6_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest6.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest6()
    return _extract_and_transpose_weights(list(model.children()))


def load_test7_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest7.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest7()
    return _extract_and_transpose_weights(list(model.children()))


def load_test8_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest8.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest8()
    return _extract_weights(model)


def load_test9_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest9.

    Parameters
    ----------
    size: int
        The size of the input data.
    patch: int
        kernel split size of the input data.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest9(size=size, patch=patch)
    return _extract_weights(model)


def load_test10_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest10.

    Parameters
    ----------
    size: int
        The size of the input data.
    patch: int
        kernel split size of the input data.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest10(size=size, patch=patch)
    return _extract_attention_weights(model=model, nb_heads=1)


def load_test11_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest11.

    Parameters
    ----------
    size: int
        The size of the input data.
    patch: int
        kernel split size of the input data.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest11(size=size, patch=patch)
    return _extract_attention_weights(model=model, nb_heads=3)


def load_test12_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTest12.

    Parameters
    ----------
    size: int
        The size of the input data.
    patch: int
        kernel split size of the input data.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTest12(size=size, patch=patch)
    return _extract_weights(model)
