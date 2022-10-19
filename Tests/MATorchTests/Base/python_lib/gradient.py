import torch
import numpy as np
from typing import Optional, List
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from python_lib.model import ModelTest1, ModelTest2


class GetGradient:
    """
    Hook that enable to get one module's gradient during the backward pass.

    Parameters
    ----------
    module: torch.nn.Module
        The module we want to get the gradient during backward.
    """

    def __init__(self, module: torch.nn.Module):
        self._hook = module.register_hook(self._get_gradient_norm)
        self.gradient_norm: Optional[float] = None

    def _get_gradient_norm(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Function hook called during the backward pass.

        Parameters
        ----------
        grad: torch.Tensor
            The gradient tensor.

        Returns
        -------
        _: Tensor
            The gradient tensor.
        """
        gradient_norm = float(torch.norm(grad).cpu().numpy())
        self.gradient_norm = gradient_norm
        return grad

    def close(self):
        """
        Remove the hook function.
        """
        self._hook.remove()


def _build_input_data(size: int) -> np.ndarray:
    """
    Build data image.

    Parameters
    ----------
    size: int
        The size of the image to build.

    Returns
    -------
    _: np.ndarray
        The image with 3 channels.
    """
    img_array = np.zeros((size, size, 3))
    for depth in range(3):
        for row in range(size):
            if depth == 0:
                img_array[row, :, depth] = \
                    (np.arange(0, size, 1) + row) / (2 * size)
            elif depth == 1:
                img_array[row, :, depth] = \
                    (np.arange(size - 1, -1, -1) + row) / (2 * size)
            else:
                img_array[row, :, depth] = \
                    (np.arange(0, size, 1) + size - 1 - row) / (2 * size)
    return img_array


def get_input_data(size: int) -> List[float]:
    """
    Get data image and flatten it.

    Parameters
    ----------
    size: int
        The size of the image to build.

    Returns
    -------
    _: List[float]
        The image with 3 channels flattened.
    """
    data: List[float] = _build_input_data(size).flatten().tolist()
    return data


def _compute_grad_norm(model: torch.nn.Module, size: int) -> float:
    """
    Compute gradient norm of one backward pass in a specific context.

    Parameters
    ----------
    model: torch.nn.Module
        The model to test.
    size: int
        The size of the input data.

    Returns
    -------
    _: float
        The gradient norm.
    """
    img_array = _build_input_data(size)
    img_tensor = ToTensor()(img_array).type(torch.float32)
    img_var = Variable(img_tensor, requires_grad=True)
    gradient = GetGradient(img_var)

    x = img_var
    x = x[None, :]
    x = model(x)

    x = x[0, 0].mean()
    loss = -1.0 / 2.0 * x * x
    loss.backward()

    gradient_norm = gradient.gradient_norm
    gradient.close()
    return gradient_norm


def compute_test1_grad_norm(size: int) -> float:
    """
    Compute gradient norm of one backward pass of ModelTest1.

    Parameters
    ----------
    size: int
        The size of the input data.

    Returns
    -------
    _: float
        The gradient norm.
    """
    torch.manual_seed(42)
    model = ModelTest1().eval().cpu()
    return _compute_grad_norm(model, size)


def compute_test2_grad_norm(size: int) -> float:
    """
    Compute gradient norm of one backward pass of ModelTest2.

    Parameters
    ----------
    size: int
        The size of the input data.

    Returns
    -------
    _: float
        The gradient norm.
    """
    torch.manual_seed(42)
    model = ModelTest2().eval().cpu()
    return _compute_grad_norm(model, size)
