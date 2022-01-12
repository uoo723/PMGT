"""
Created on 2022/01/06
@author Sangwoo Han
"""
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def load_node_init_emb(
    item_encoder_path: str, node_encoder_path: str, node_init_emb_path: str
) -> np.ndarray:
    item_encoder: MultiLabelBinarizer = joblib.load(item_encoder_path)
    node_encoder: MultiLabelBinarizer = joblib.load(node_encoder_path)
    node_emb_init: np.ndarray = np.load(node_init_emb_path)

    item2idx = {item: i for i, item in enumerate(node_encoder.classes_)}

    item_emb_init = np.empty(
        (len(item_encoder.classes_), node_emb_init.shape[1]), dtype=node_emb_init.dtype
    )

    for i, item in enumerate(item_encoder.classes_):
        if item in item2idx:
            item_emb_init[i] = node_emb_init[item2idx[item]]
        else:
            item_emb_init[i] = np.random.normal(size=node_emb_init.shape[1])

    return item_emb_init
