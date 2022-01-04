"""
Created on 2022/01/04
@author Sangwoo Han
"""
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


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
