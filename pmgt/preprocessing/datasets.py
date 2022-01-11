"""
Created on 2022/01/11
@author Sangwoo Han
"""
import os
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class AmazonReviewImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, transforms: Callable, item_ids: Optional[np.ndarray] = None
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.item_ids = item_ids
        self.images, self._num_images = self._get_image_list()

    @property
    def num_images(self) -> Dict[str, int]:
        return self._num_images

    def _get_image_list(self) -> Tuple[List[str], Dict[str, int]]:
        images = []
        num_images = OrderedDict()
        for item_id in os.listdir(self.root):
            if self.item_ids is not None and item_id not in self.item_ids:
                continue
            imagefile_list = os.listdir(os.path.join(self.root, item_id))
            num_images[item_id] = len(imagefile_list)
            for image_name in imagefile_list:
                images.append(os.path.join(item_id, image_name))
        return images, num_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(os.path.join(self.root, self.images[idx])).convert("RGB")
        return self.transforms(img)

    def __len__(self) -> int:
        return len(self.images)


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
