import torch
from pathlib import Path

from python_lib.nlp.generate import generate_with_cache
from python_lib.nlp.model import Transformer, TransformerArgs
from python_lib.nlp.llama3.tokenizer import Tokenizer, ChatFormat


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
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    formatter = ChatFormat(tokenizer)

    model_args = TransformerArgs(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=10000
    )

    model = Transformer(model_args)
    model.load_state_dict(state)
    model.to("mps")

    print(prompt, end="", flush=True)
    dialogs = [
        [
            {"role": "user", "content": prompt},
        ],
    ]
    prompt = torch.tensor(
        [
            formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ][0],
        dtype=torch.long, device="mps"
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


if __name__ == "__main__":
    model_path = "/Users/jean-francoisreboud/DocumentsNonSync/Projet/Python/mistral/weights/Meta-Llama-3-8B-Instruct/"
    prompt = "How do you do?"

    generate(
        prompt="How do you do?",
        model_path=model_path,
        max_tokens=1000,
    )
