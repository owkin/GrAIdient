import os
from pathlib import Path
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer = [
        f for f in os.listdir(Path(model_path))
        if f.startswith("tokenizer.model")
    ]
    assert (
        len(tokenizer) > 0
    ), f"No tokenizer found in {model_path}, " \
       f"make sure to place a `tokenizer.model.[v1,v2,v3]` file " \
       f"in {model_path}."
    assert (
        len(tokenizer) == 1
    ), f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, " \
       f"make sure to only have one tokenizer"

    tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer[0]))

    return tokenizer
