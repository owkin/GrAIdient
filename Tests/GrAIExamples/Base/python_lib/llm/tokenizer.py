from typing import List
from pathlib import Path
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """
    Tokenizer to encode / decode into tokens.

    Parameters
    ----------
    model_path: str
        The path to the weights of the tokenizer on the disk.
    """

    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        """
        End of sequence token.
        """
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        """
        Padding token.
        """
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        """
        Encode a prompt into a sequence of tokens.

        Parameters
        ----------
        s: str
            The input prompt.

        Returns
        -------
        _: [int]
            The output sequence of tokens.
        """
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        """
        Decode a sequence of tokens into prompt.

        Parameters
        ----------
        t: [int]
            The input sequence of tokens.

        Returns
        -------
        _: [int]
            The output prompt.
        """
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out
