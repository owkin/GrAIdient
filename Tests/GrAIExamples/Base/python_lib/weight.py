import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from safetensors.torch import load_file
from python_lib.model import SimpleAutoEncoder


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
        print(f"Extracting weights {name}.")
        weights_list, dims_list = _flatten_weights(
            layer_weights.data.cpu().float().numpy()
        )

        layers_weights.append(weights_list)
        layers_dims.append(dims_list)

    return layers_weights, layers_dims


def _extract_state(
    state: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """
    Get weights and biases.

    Parameters
    ----------
    state: [str: torch.Tensor]
        The module state, containing the weights and biases.

    Returns
    -------
    layer_weights: Dict[str, np.ndarray]
        Dictionary of flattened weights.
    """
    layers_weights: Dict[str, np.ndarray] = {}
    for name, layer_weights in state.items():
        print(f"Extracting weights {name}.")
        weights_list, _ = _flatten_weights(
            layer_weights.data.cpu().float().numpy()
        )
        layers_weights[name] = weights_list
    return layers_weights


def extract_state_key(
    key: str,
    state: Dict[str, torch.Tensor]
) -> np.ndarray:
    """
    Get weights and biases.

    Parameters
    ----------
    key: str
        Key to extract.
    state: [str: torch.Tensor]
        The module state, containing the weights and biases.

    Returns
    -------
    weights_list: np.ndarray
        Array of flattened weights.
    """
    print(f"Extracting weigths {key}.")
    weights_list, _ = _flatten_weights(
        state[key].data.cpu().float().numpy()
    )
    return weights_list


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


def load_mistral_weights(
    model_path: str
) -> Dict[str, np.ndarray]:
    """
    Get weights and biases for Mistral-7B-Instruct-v0.3 LLM.

    Returns
    -------
    _: Dict[str, np.ndarray]
        Dictionary of flattened weights.
    """
    state = load_file(
        str(Path(model_path) / "consolidated.safetensors"),
        "cpu"
    )
    return _extract_state(state)


def load_llama_state(
    model_path: str
) -> Dict[str, torch.Tensor]:
    """
    Get state for Llama-2-7B-Chat or Llama-3-8B-Instruct.

    Returns
    -------
    _: Dict[str, np.ndarray]
        Dictionary of flattened weights.
    """
    state = torch.load(
        str(Path(model_path) / "consolidated.00.pth"),
        "cpu"
    )
    return state


if __name__ == "__main__":
    state = load_llama_state("/Users/jean-francoisreboud/DocumentsNonSync/Projet/Python/mistral/weights/llama-2-7b-chat/")
    test = extract_state_key("tok_embeddings.weight", state)
    print("COUCOU")

