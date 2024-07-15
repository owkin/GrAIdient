import time
import torch
from typing import List
from pathlib import Path

from python_lib.nlp.llama2.tokenizer import Tokenizer
from python_lib.nlp.generate import generate_with_cache
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
    state = torch.load(str(Path(model_path) / "consolidated.00.pth"))
    state.pop("rope.freqs")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    print(prompt)
    prompt = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device="mps"
    )

    model_args = TransformerArgs(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=11008,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=1e-5,
        vocab_size=32000,
        rope_theta=10000
    )

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
        if token == tokenizer.eos_id:
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


def load_llama2_tokenizer(model_path: str) -> Tokenizer:
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
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    return tokenizer


def encode_llama2(
    prompt: str,
    tokenizer: Tokenizer
) -> List[int]:
    """
    Encode text.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    tokenizer: Tokenizer
        The tokenizer.

    Returns
    -------
    _: List of encoded tokens.
    """
    return tokenizer.encode(prompt)


def decode_llama2(
    prompt: List[int],
    tokenizer: Tokenizer
) -> str:
    """
    Decode text.

    Parameters
    ----------
    prompt: [int]
        The input prompt.
    tokenizer: Tokenizer
        The tokenizer.

    Returns
    -------
    _: Decoded text.
    """
    return tokenizer.decode(prompt)


if __name__ == "__main__":
    model_path = "/TO/UPDATE/llama-2-7b-chat/"
    prompt = "How do you do?"

    generate(
        prompt=prompt,
        model_path=model_path,
        max_tokens=4096,
    )
