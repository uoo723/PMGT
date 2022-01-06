"""
Created on 2022/01/06
@author Sangwoo Han
@ref https://github.com/guoyang9/NCF/blob/master/data_utils.py
"""
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

TFeatures = List[Tuple[int, int]]


class NCFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features: TFeatures,
        num_user: int,
        num_item: int,
        num_ng: int = 0,
        is_training: bool = True,
    ) -> None:
        super().__init__()
        """ Note that the labels are only useful when training, we thus
			add them in the ng_sample() function.
		"""
        self._features = self.features = features
        self.labels = np.ones(len(features), dtype=np.float32)
        self.num_user = num_user
        self.num_item = num_item
        self.num_ng = num_ng
        self.is_training = is_training
        self.user_item_mat, self.users, self.items = self._build_mat(features)
        self._gt = None
        self._mlb = None

        if not self.is_training:
            self.test_data = self._build_test_data()

    @property
    def gt(self) -> csr_matrix:
        assert not self.is_training, "gt is only accessible, when testing"
        return self._gt

    @property
    def mlb(self) -> MultiLabelBinarizer:
        assert not self.is_training, "mlb is only accessible, when testing"
        return self._mlb

    def _build_mat(
        self, features: TFeatures
    ) -> Tuple[sp.dok_matrix, np.ndarray, np.ndarray]:
        mat = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)
        users = []
        items = []
        for u, i in features:
            users.append(u)
            items.append(i)
            mat[u, i] = 1.0
        return mat, np.asarray(users), np.asarray(items)

    def _build_test_data(self) -> List[Tuple[int, List[int]]]:
        mat = self.user_item_mat.tocsr()
        indices = self.users.argsort()
        users = self.users[indices]
        items = self.items[indices]
        item_nums = mat[np.unique(self.users[indices])].sum(axis=-1, dtype=np.int32).A1

        s = 0
        test_data = []
        gt = []
        for i, num in enumerate(item_nums):
            e = s + num
            test_data.append((users[i], items[s:e].tolist()))
            gt.append(items[s:e].copy())
            s = e

        self._mlb = MultiLabelBinarizer(
            sparse_output=True, classes=np.arange(self.num_item)
        )
        self._gt = self._mlb.fit_transform(gt)

        return test_data

    def ng_sample(self) -> None:
        assert self.is_training, "no need to sampling when testing"

        features_ng = []
        for x in self.features:
            u = x[0]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.user_item_mat:
                    j = np.random.randint(self.num_item)
                features_ng.append([u, j])

        labels_ps = np.ones(len(self.features), dtype=np.float32)
        labels_ng = np.zeros(len(features_ng), dtype=np.float32)

        self.features = self._features + features_ng
        self.labels = np.concatenate([labels_ps, labels_ng])

    def __len__(self) -> int:
        return len(self.labels) if self.is_training else len(self.test_data)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[int, int, float], Tuple[int, np.ndarray, np.ndarray]]:
        if self.is_training:
            user = self.features[idx][0]
            item = self.features[idx][1]
            label = self.labels[idx]
            return user, item, label

        user, items = self.test_data[idx]
        candidate_items = items.copy()
        labels = [1] * len(items)
        for _ in range(self.num_ng - len(items)):
            neg_item = np.random.randint(self.num_item)
            while (user, neg_item) in self.user_item_mat:
                neg_item = np.random.randint(self.num_item)
            candidate_items.append(neg_item)
            labels.append(0)

        return (
            user,
            np.asarray(candidate_items, dtype=np.int64),
            np.asarray(labels, dtype=np.float32),
        )
