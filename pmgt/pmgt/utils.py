"""
Created on 2022/01/06
@author Sangwoo Han
"""
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize as sk_normalize


def load_node_init_emb(
    item_encoder_path: str,
    node_encoder_path: str,
    node_init_emb_path: str,
    normalize: bool = True,
) -> np.ndarray:
    item_encoder: MultiLabelBinarizer = joblib.load(item_encoder_path)
    node_encoder: MultiLabelBinarizer = joblib.load(node_encoder_path)
    node_init_emb: np.ndarray = np.load(node_init_emb_path)

    item2idx = {item: i for i, item in enumerate(node_encoder.classes_)}

    item_init_emb = np.empty(
        (len(item_encoder.classes_), node_init_emb.shape[1]), dtype=node_init_emb.dtype
    )

    for i, item in enumerate(item_encoder.classes_):
        if item in item2idx:
            item_init_emb[i] = node_init_emb[item2idx[item]]
        else:
            item_init_emb[i] = np.random.normal(size=node_init_emb.shape[1])

    if normalize:
        item_init_emb = sk_normalize(item_init_emb)

    return item_init_emb


def get_input_feat_embeds(
    node_ids: torch.LongTensor, feat_embeddings_list: nn.ModuleList
) -> Tuple[torch.FloatTensor]:
    input_feat_embeds = [
        feat_embeddings(node_ids) for feat_embeddings in feat_embeddings_list
    ]

    return input_feat_embeds
