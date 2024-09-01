import time
import torch
from typing import List
from pathlib import Path

from safetensors.torch import load_file
from python_lib.nlp.gemma2.tokenizer import Tokenizer
from python_lib.nlp.generate import generate_with_cache
from python_lib.nlp.gemma2.model import Transformer, TransformerArgs


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
    state1 = load_file(
        str(Path(model_path) / "model-00001-of-00002.safetensors"),
    )
    state2 = load_file(
        str(Path(model_path) / "model-00002-of-00002.safetensors"),
    )

    state = state1
    state.update(state2)
    state["model.output.weight"] = state["model.embed_tokens.weight"]

    state_copy = {}
    for key, value in state.items():
        new_key = key.replace("model.", "")
        state_copy[new_key] = value
    state = state_copy

    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    print(prompt)
    prompt = torch.tensor(
        [2, 106] +
        tokenizer.encode("user", bos=False) +
        tokenizer.encode(prompt, bos=False) +
        [107, 106] +
        tokenizer.encode("model", bos=False),
        dtype=torch.long, device="mps"
    )

    model_args = TransformerArgs(
        dim=2304,
        n_layers=26,
        head_dim=256,
        hidden_dim=9216,
        n_heads=8,
        n_kv_heads=4,
        norm_eps=1e-6,
        vocab_size=256000,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
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
        if token == 107 or token == 1:
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


def load_gemma2_tokenizer(model_path: str) -> Tokenizer:
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


def encode_gemma2(
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


def decode_gemma2(
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
    model_path = "/TO/UPDATE/gemma-2-2b-it/"
    prompt = "What is the meaning of life?"

    generate(
        prompt=prompt,
        model_path=model_path,
        temp=0,
        max_tokens=4096,
    )
