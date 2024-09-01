import torch
from typing import Generator, Optional

from python_lib.nlp.model import Transformer


def predict_no_cache(
    prompt: torch.Tensor,
    model: Transformer,
    temp: float = 0.0,
    n_layers: Optional[int] = None
) -> torch.Tensor:
    """
    Predict text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model: Transformer
        The model to use for generation.
    temp: float
        The temperature for sampling. If temp is 0, use max sampling.
    n_layers: int
        Modifier of the number of Transformer blocks.

    Returns
    -------
    y: torch.Tensor
        The generated text.
    """
    def sample(logits: torch.Tensor) -> torch.Tensor:
        return (
            torch.argmax(logits, dim=-1)
            if temp == 0
            else torch.multinomial(
                torch.softmax(logits, dim=-1) * (1 / temp), 1
            )
        )

    y = prompt
    logits, _ = model(y[None], cache=None, n_layers=n_layers)
    return sample(logits)


def generate_with_cache(
    prompt: torch.Tensor, model: Transformer, temp: float = 0.0
) -> Generator[torch.Tensor, None, None]:
    """
    Generate text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model: Transformer
        The model to use for generation.
    temp: float
        The temperature for sampling. If temp is 0, use max sampling.

    Returns
    -------
    y: torch.Tensor
        The generated text.
    """
    def sample(logits: torch.Tensor) -> torch.Tensor:
        return (
            torch.argmax(logits, dim=-1)
            if temp == 0
            else torch.multinomial(
                torch.softmax(logits, dim=-1) * (1 / temp), 1
            )[0]
        )

    y = prompt
    cache = None

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y
