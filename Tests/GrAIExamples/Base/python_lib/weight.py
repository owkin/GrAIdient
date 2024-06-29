import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from python_lib.model import SimpleAutoEncoder
from python_lib.nlp.model import Transformer, TransformerArgs


def _flatten_weights(
    weights: np.ndarray
) -> Tuple[np.ndarray, List[int]]:
    """
    Flatten weights and biases.

    Parameters
    ----------
    weights: np.ndarray
        The weights to flatten.

    Returns
    -------
    (_, _): np.ndarray, List[int]
        The flattened weights, their shape.
    """
    weights_list = weights.flatten()
    dims_list = list(weights.shape)

    return weights_list, dims_list


def _extract_weights(
    state: Dict[str, torch.Tensor]
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Get weights and biases.

    Parameters
    ----------
    state: [str: torch.Tensor]
        The module state, containing the weights and biases.

    Returns
    -------
    (_, _): List[np.ndarray], List[List[int]]
        The flattened weights, their shape.
    """
    layers_weights: List[np.ndarray] = []
    layers_dims: List[List[int]] = []
    for name, layer_weights in state.items():
        print(f"Extracting weigths {name}.")
        weights_list, dims_list = _flatten_weights(
            layer_weights.data.cpu().float().numpy()
        )

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)

    return layers_weights, layers_dims


def _extract_and_transpose_weights(
    modules: [torch.nn.Module]
) -> Tuple[List[np.ndarray], List[List[int]]]:
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
    (_, _): List[np.ndarray], List[List[int]]
        The flattened weights, their shape.
    """
    layers_weights: List[np.ndarray] = []
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
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Get weights and biases for simple auto encoder model.

    Returns
    -------
    (_, _): List[np.ndarray], List[List[int]]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model = SimpleAutoEncoder()
    return _extract_and_transpose_weights(list(model.children()))


def load_llm_weights(
    model_path: str
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Get weights and biases for LLM.

    Returns
    -------
    (_, _): List[np.ndarray], List[List[int]]
        The flattened weights, their shape.
    """
    torch.manual_seed(42)
    model_args = TransformerArgs(
        dim=2,
        n_layers=32,
        head_dim=2,
        hidden_dim=2,
        n_heads=2,
        n_kv_heads=1,
        norm_eps=1e-5,
        vocab_size=32000
    )
    model = Transformer(model_args)
    state = model.state_dict()
    """state = torch.load(
        str(Path(model_path) / "consolidated.00.pth"),
        map_location="cpu"
    )"""
    return _extract_weights(state)
