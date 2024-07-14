import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from safetensors.torch import load_file

from python_lib.nlp.generate import (
    predict_no_cache,
    generate_with_cache
)
from python_lib.nlp.mistral.tokenizer import load_tokenizer
from mistral_common.tokens.tokenizers.base import Tokenizer
from python_lib.nlp.model import Transformer, TransformerArgs


def generate(
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
    state = load_file(str(Path(model_path) / "consolidated.safetensors"))
    mistral_tokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)
        model_args.rope_theta = 10000

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    print(prompt, end="", flush=True)
    prompt = torch.tensor(
        tokenizer.encode(prompt, bos=True, eos=False),
        dtype=torch.long,
        device="mps"
    )

    tokens = []
    skip = 0
    for token, n in zip(
        generate_with_cache(prompt, model, temp),
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
    n_layers: Optional[int] = None
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
    n_layers: int
        Modifier of the number of Transformer blocks.
    """
    state = load_file(str(Path(model_path) / "consolidated.safetensors"))
    mistral_tokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

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
        tokenizer.encode(prompt, bos=True, eos=False),
        dtype=torch.long,
        device="mps"
    )

    tokens = predict_no_cache(
        prompt, model, temp, n_layers
    ).squeeze(dim=0).cpu().numpy().tolist()

    prediction = tokenizer.decode(tokens)
    print(prediction)


def predict(
    prompt: str,
    model_path: str,
    n_layers: Optional[int] = None
) -> np.ndarray:
    """
    Predict text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model_path: str
        Path to the model on the disk.
    n_layers: int
        Modifier of the number of Transformer blocks.
    """
    state = load_file(str(Path(model_path) / "consolidated.safetensors"))
    mistral_tokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    prompt = torch.tensor(
        tokenizer.encode(prompt, bos=True, eos=False),
        dtype=torch.long,
        device="mps"
    )
    out, _ = model(prompt[None], n_layers=n_layers)
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
    mistral_tokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    return tokenizer.encode(prompt, bos=True, eos=False)


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
    mistral_tokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    return tokenizer.decode(prompt)


if __name__ == "__main__":
    model_path = "/Users/jean-francoisreboud/DocumentsNonSync/Projet/Python/mistral/weights/mistral-7B-v0.3/"
    prompt = "How do you do?"

    generate(
        prompt="How do you do?",
        model_path=model_path,
        max_tokens=1000,
    )
    prompt = encode(
        prompt=prompt,
        model_path=model_path
    )
    prompt = decode(
        prompt=prompt,
        model_path=model_path
    )
    _predict(
        prompt=prompt,
        model_path=model_path,
        n_layers=None
    )
    predict(
        prompt=prompt,
        model_path=model_path,
        n_layers=1
    )
