import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from safetensors.torch import load_file

from python_lib.nlp.generate import (
    predict_no_cache,
    generate_with_cache
)
from python_lib.nlp.model import Transformer, TransformerArgs
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest


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
    tokenizer = MistralTokenizer.from_file(
        str(Path(model_path) / "tokenizer.model.v3")
    )

    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=prompt),
        ],
    )
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    print(prompt)
    prompt = torch.tensor(tokens, dtype=torch.long, device="mps")

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)
        model_args.rope_theta = 10000

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    start_time = time.time()
    print("Start generating...")

    tokens = []
    skip = 0
    for token, n in zip(
        generate_with_cache(prompt, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.instruct_tokenizer.tokenizer.eos_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1

    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("End generating.")

    if len(tokens) == 0:
        print("No tokens generated for this prompt.")
        return

    elapsed_time = time.time() - start_time
    print(f"Generation took: {elapsed_time:.6f} seconds.")


def predict(
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
    tokenizer = MistralTokenizer.from_file(
        str(Path(model_path) / "tokenizer.model.v3")
    )

    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=prompt),
        ],
    )
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    print(prompt)
    prompt = torch.tensor(tokens, dtype=torch.long, device="mps")

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)
        model_args.rope_theta = 10000

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    tokens = predict_no_cache(
        prompt, model, temp, n_layers
    ).squeeze(dim=0).cpu().numpy().tolist()

    prediction = tokenizer.decode(tokens)
    print(prediction)


def predict_mistral(
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
    tokenizer = MistralTokenizer.from_file(
        str(Path(model_path) / "tokenizer.model.v3")
    )

    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=prompt),
        ],
    )
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    prompt = torch.tensor(tokens, dtype=torch.long, device="mps")

    with open(Path(model_path) / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        model_args = TransformerArgs(**config)
        model_args.rope_theta = 10000

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    out, _ = model(prompt[None], n_layers=n_layers)
    return out.detach().cpu().numpy().flatten()


def load_mistral_tokenizer(model_path: str) -> MistralTokenizer:
    """
    Load tokenizer from the disk.

    Parameters
    ----------
    model_path: str
        Path to the model on the disk.

    Returns
    -------
    tokenizer: Tokenizer
        The loaded tokenizer.
    """
    tokenizer = MistralTokenizer.from_file(
        str(Path(model_path) / "tokenizer.model.v3")
    )
    return tokenizer


def encode_mistral(
    prompt: str,
    tokenizer: MistralTokenizer
) -> List[int]:
    """
    Encode text.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    tokenizer: MistralTokenizer
        The tokenizer.

    Returns
    -------
    _: List of encoded tokens.
    """
    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=prompt),
        ],
    )
    return tokenizer.encode_chat_completion(completion_request).tokens


def decode_mistral(
    prompt: List[int],
    tokenizer: MistralTokenizer
) -> str:
    """
    Decode text.

    Parameters
    ----------
    prompt: [int]
        The input prompt.
    tokenizer: MistralTokenizer
        The tokenizer.

    Returns
    -------
    _: Decoded text.
    """
    return tokenizer.decode(prompt)


if __name__ == "__main__":
    model_path = "/TO/UPDATE/mistral-7B-Instruct-v0.3/"
    prompt = "What is the meaning of life?"

    generate(
        prompt=prompt,
        model_path=model_path,
        max_tokens=4096,
    )

    """
    prompt = "How do you do?"
    tokenizer = load_tokenizer(model_path)
    prompt = encode(
        prompt=prompt,
        tokenizer=tokenizer
    )
    prompt = decode(
        prompt=prompt,
        tokenizer=tokenizer
    )
    _redict(
        prompt=prompt,
        model_path=model_path,
        n_layers=None
    )
    predict_mistral(
        prompt=prompt,
        model_path=model_path,
        n_layers=1
    )
    """
