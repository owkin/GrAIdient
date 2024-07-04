import json
import torch
import numpy as np
from pathlib import Path
from typing import Generator, List

from python_lib.nlp.tokenizer import Tokenizer
from python_lib.nlp.model import Transformer, TransformerArgs


def _predict_no_cache(
    prompt: torch.Tensor, model: Transformer, temp: float = 0.0
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
    logits, _ = model(y[None], cache=None)
    return sample(logits)


def _generate_with_cache(
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
            )
        )

    y = prompt
    cache = None

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y


def _generate(
    prompt: str,
    model_path: str,
    temp: float = 0,
    max_tokens: int = 128
):
    """
    Generate text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model_path: str
        Path to the model on the disk.
    temp: float
        The temperature for sampling. If temp is 0, use max sampling.
    max_tokens: int
        The maximal number of generated tokens.
    """
    state = torch.load(str(Path(model_path) / "consolidated.00.pth"))
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    print(prompt, end="", flush=True)
    prompt = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device="mps"
    )

    tokens = []
    skip = 0
    for token, n in zip(
        _generate_with_cache(prompt, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.eos_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1

    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)

    if len(tokens) == 0:
        print("No tokens generated for this prompt.")
        return


def _predict(
    prompt: str,
    model_path: str,
    temp: float = 0,
):
    """
    Predict text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model_path: str
        Path to the model on the disk.
    temp: float
        The temperature for sampling. If temp is 0, use max sampling.
    """
    state = torch.load(str(Path(model_path) / "consolidated.00.pth"))
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    print(prompt, end="", flush=True)
    prompt = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device="mps"
    )

    tokens = _predict_no_cache(
        prompt, model, temp
    ).squeeze(dim=0).cpu().numpy().tolist()
    print(tokenizer.decode(tokens))


def predict(
    prompt: str,
    model_path: str
) -> np.ndarray:
    """
    Predict text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model_path: str
        Path to the model on the disk.
    """
    state = torch.load(str(Path(model_path) / "consolidated.00.pth"))
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    prompt = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device="mps"
    )
    out, _ = model(prompt[None])
    return out.detach().cpu().numpy().flatten()


def encode(
    prompt: str,
    model_path: str
) -> List[int]:
    """
    Encode text.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model_path: str
        Path to the model on the disk.
    """
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    return tokenizer.encode(prompt)


def decode(
    prompt: List[int],
    model_path: str
) -> str:
    """
    Decode text.

    Parameters
    ----------
    prompt: [int]
        The input prompt.
    model_path: str
        Path to the model on the disk.
    """
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    return tokenizer.decode(prompt)


if __name__ == "__main__":
    model_path = ""
    prompt = encode(
        prompt="How do you do?",
        model_path=model_path
    )
    prompt = decode(
        prompt=prompt,
        model_path=model_path
    )
    _predict(
        prompt="How do you do?",
        model_path=model_path,
    )
    predict(
        prompt="How do you do?",
        model_path=model_path
    )
