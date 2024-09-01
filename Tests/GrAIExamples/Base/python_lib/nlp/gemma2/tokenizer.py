# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List, Optional

import sentencepiece


class Tokenizer:
    """
    Tokenizer to encode / decode into tokens.

    Parameters
    ----------
    model_path: str
        The path to the weights of the tokenizer on the disk.
    """

    def __init__(self, model_path: Optional[str]):
        # Reload tokenizer.
        assert os.path.isfile(model_path), model_path
        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
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
        assert isinstance(s, str)
        t = self.sp_model.EncodeAsIds(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

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
        return self.sp_model.DecodeIds(t)
