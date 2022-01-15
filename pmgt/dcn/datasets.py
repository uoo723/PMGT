"""
Created on 2022/01/15
@author Sangwoo Han
"""
from typing import List, Tuple, Union

import numpy as np

from ..ncf.datasets import NCFDataset


class DCNDataset(NCFDataset):
    @property
    def gt(self) -> np.ndarray:
        assert not self.is_training, "gt is only accessible, when testing"
        return self._gt

    def _build_test_data(self) -> List[Tuple[int, List[int]]]:
        self._gt = np.ones(len(self._features), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels) if self.is_training else len(self._features)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[int, int, float], Tuple[int, np.ndarray, np.ndarray]]:
        features = self.features if self.is_training else self._features
        user = features[idx][0]
        item = features[idx][1]
        label = self.labels[idx] if self.is_training else 1
        return (user, item), label
