import torch
import numpy as np
from typing import List, Tuple

from python_lib.model import (
    ModelTestConv1,
    ModelTestConv2,
    ModelTestConvSK,
    ModelTestDeConvSK,
    ModelTestCat,
    ModelTestResize,
    ModelTestPatchConv,
    ModelTestAttention1,
    ModelTestAttention2,
    ModelTestLayerNorm,
    ModelTestAutoEncoder1,
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

    cur_item = 0
    list_items = list(model_weights.items())

    while cur_item < len(list_items):
        name, layer_weights = list_items[cur_item]
        print(f"Extracting weigths {name}.")

        if "in_proj" in name:
            weights = layer_weights.data.cpu().numpy()
            nb_partial = int(len(weights) / 3)

            weights1 = weights[0: nb_partial]
            weights2 = weights[nb_partial: 2*nb_partial]
            weights3 = weights[2*nb_partial: 3*nb_partial]

            cur_item += 1
            name, layer_weights = list_items[cur_item]
            print(f"Extracting weigths {name}.")
            biases = layer_weights.data.cpu().numpy()

            biases1 = biases[0: nb_partial]
            biases2 = biases[nb_partial: 2 * nb_partial]
            biases3 = biases[2 * nb_partial: 3 * nb_partial]

            weights_list, dims_list = _flatten_weights(
                weights=weights1
            )
            layers_weights.append(weights_list)
            layers_dims.append(dims_list)
            weights_list, dims_list = _flatten_weights(
                weights=biases1
            )
            layers_weights.append(weights_list)
            layers_dims.append(dims_list)

            weights_list, dims_list = _flatten_weights(
                weights=weights2
            )
            layers_weights.append(weights_list)
            layers_dims.append(dims_list)
            weights_list, dims_list = _flatten_weights(
                weights=biases2
            )
            layers_weights.append(weights_list)
            layers_dims.append(dims_list)

            weights_list, dims_list = _flatten_weights(
                weights=weights3
            )
            layers_weights.append(weights_list)
            layers_dims.append(dims_list)
            weights_list, dims_list = _flatten_weights(
                weights=biases3
            )
            layers_weights.append(weights_list)
            layers_dims.append(dims_list)

            cur_item += 1

        else:
            weights_list, dims_list = _flatten_weights(
                layer_weights.data.cpu().numpy()
            )

            layers_weights.append(weights_list)
            layers_dims.append(dims_list)

            cur_item += 1

    return layers_weights, layers_dims


def load_conv1_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestConv1.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestConv1()
    return _extract_weights(model)


def load_conv2_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestConv2.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestConv2()
    return _extract_weights(model)


def load_conv_sk_weights(
    stride: int, kernel: int
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestConvSK.

    Parameters
    ----------
    stride: int
        The stride of the model.
    kernel: int
        The kernel size of the model.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestConvSK(stride=stride, kernel=kernel)
    return _extract_weights(model)


def load_deconv_sk_weights(
    stride: int, kernel: int
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestDeConvSK.

    Parameters
    ----------
    stride: int
        The stride of the model.
    kernel: int
        The kernel size of the model.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestDeConvSK(stride=stride, kernel=kernel)
    return _extract_and_transpose_weights(list(model.children()))


def load_cat_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestCat.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestCat()
    return _extract_weights(model)


def load_resize_weights(size: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestResize.

    Parameters
    ----------
    size: int
        The output size of the resize operation.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestResize(size)
    return _extract_weights(model)


def load_patch_conv_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestPatchConv.

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
    model = ModelTestPatchConv(size=size, patch=patch)
    return _extract_weights(model)


def load_attention1_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestAttention1.

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
    model = ModelTestAttention1(size=size, patch=patch)
    return _extract_attention_weights(model=model)


def load_attention2_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestAttention2.

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
    model = ModelTestAttention2(size=size, patch=patch)
    return _extract_attention_weights(model=model)


def load_layer_norm_weights(size: int, patch: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestLayerNorm.

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
    model = ModelTestLayerNorm(size=size, patch=patch)
    return _extract_weights(model)


def load_auto_encoder1_weights() -> Tuple[List[List[float]], List[List[int]]]:
    """
    Get weights and biases for ModelTestAutoEncoder1.

    Returns
    -------
    (_, _): List[float], List[int]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = ModelTestAutoEncoder1()
    return _extract_and_transpose_weights(list(model.children()))
