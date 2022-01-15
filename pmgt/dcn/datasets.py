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
        return self.labels

    def _build_test_data(self) -> List[Tuple[int, List[int]]]:
        """Do nothing"""

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[int, int, float], Tuple[int, np.ndarray, np.ndarray]]:
        user = self.features[idx][0]
        item = self.features[idx][1]
        label = self.labels[idx]
        return (user, item), label
