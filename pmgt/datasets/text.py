"""
Created on 2022/01/04
@author Sangwoo Han
"""
from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class AmazonReviewTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: Dict[str, np.ndarray]) -> None:
        self.texts, self._num_texts = self._get_text_list(texts)

    @property
    def num_texts(self) -> Dict[str, int]:
        return self._num_texts

    def _get_text_list(
        self, item_texts: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        texts = []
        num_texts = OrderedDict()
        for item_id, text_array in item_texts.items():
            num_texts[item_id] = len(text_array)
            texts.append(text_array)
        return np.concatenate(texts), num_texts

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def __len__(self) -> int:
        return len(self.texts)


def text_collate_fn(
    batch: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        batch,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_tensors="pt",
    )
