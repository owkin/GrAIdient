import json
import torch
from pathlib import Path
from typing import Generator

from python_lib.llm.tokenizer import Tokenizer
from python_lib.llm.model import LLM, ModelArgs


def generate_with_cache(
    prompt: torch.Tensor, model: LLM, temp: float = 0.0
) -> Generator[torch.Tensor, None, None]:
    """
    Generate text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model: LLM
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

    cache = None
    y = prompt[None, ...]

    while True:
        logits, cache = model(y, cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y


def generate(
    prompt: str,
    model: LLM,
    tokenizer: Tokenizer,
    temp: float,
    max_tokens: int
):
    """
    Generate text based on the given prompt and model.

    Parameters
    ----------
    prompt: torch.Tensor
        The input prompt.
    model: LLM
        The model to use for generation.
    tokenizer: Tokenizer
        The tokenizer to encode / decode into tokens.
    temp: float
        The temperature for sampling. If temp is 0, use max sampling.
    max_tokens: int
        The maximal number of generated tokens.
    """
    print(prompt, end="", flush=True)
    prompt = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device="mps"
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
        print("No tokens generated for this prompt")
        return


if __name__ == "__main__":
    model_path = Path("TO_MODIFY/mistral/weights/mistral-7B-v0.1")
    state = torch.load(str(model_path / "consolidated.00.pth"))
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))

    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)

    model = LLM(model_args)
    model.load_state_dict(state)
    model.to("mps")

    generate(
        "Hello, what is your name?",
        model,
        tokenizer,
        0.7,
        200
    )
