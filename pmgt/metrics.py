"""
Created on 2022/01/05
@author Sangwoo Han
"""
from typing import Hashable, Iterable, Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

TPredict = np.ndarray
TTarget = Iterable[Iterable[Hashable]]
TMlb = Optional[MultiLabelBinarizer]


def get_ndcg(prediction: TPredict, targets: TTarget, mlb: TMlb = None, top=5) -> float:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((targets.shape[0], 1))
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i : i + 1])
        dcg += p.multiply(targets).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])
