"""
Created on 2022/01/13
@author Sangwoo Han
"""

from typing import Dict, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
import torch

from ..ncf.datasets import NCFDataset
from ..pmgt.datasets import get_input_tensor

TItem = Tuple[torch.LongTensor, torch.FloatTensor]


class PMGT_NCFDataset(NCFDataset):
    def __init__(
        self,
        graph: nx.Graph,
        hop_sampling_sizes: List[int],
        max_num_ctx_neigh: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph = graph
        self.hop_sampling_sizes = hop_sampling_sizes
        self.max_num_ctx_neigh = max_num_ctx_neigh

    def ng_sample(self) -> None:
        assert self.is_training, "no need to sampling when testing"

        features_ng = []
        for x in self._features:
            u = x[0]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.user_item_mat:
                    j = np.random.randint(self.num_item)
                features_ng.append([u, j])

        labels_ps = np.ones(len(self._features), dtype=np.float32)
        labels_ng = np.zeros(len(features_ng), dtype=np.float32)

        self.features = self._features + features_ng
        self.labels = np.concatenate([labels_ps, labels_ng])

    def __len__(self) -> int:
        return len(self.labels) if self.is_training else len(self.test_data)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[Tuple[int, TItem], float],
        Tuple[Tuple[int, torch.LongTensor], torch.FloatTensor],
    ]:
        if self.is_training:
            user = self.features[idx][0]
            label = self.labels[idx]
            item = get_input_tensor(
                self.graph,
                self.features[idx][1] + 2,
                self.hop_sampling_sizes,
                self.max_num_ctx_neigh,
            )
            return (user, item), label

        user, items = self.test_data[idx]
        candidate_items = items.copy()
        labels = [1] * len(items)
        for _ in range(self.num_ng - len(items)):
            neg_item = np.random.randint(self.num_item)
            while (user, neg_item) in self.user_item_mat:
                neg_item = np.random.randint(self.num_item)
            candidate_items.append(neg_item)
            labels.append(0)

        return ((user, torch.LongTensor(candidate_items)), torch.FloatTensor(labels))

        # user, items = self.test_data[idx]
        # candidate_items = []
        # for pos_item in items:
        #     candidate_items.append(
        #         get_input_tensor(
        #             self.graph,
        #             pos_item + 2,
        #             self.hop_sampling_sizes,
        #             self.max_num_ctx_neigh,
        #         )
        #     )
        # labels = [1] * len(items)
        # for _ in range(self.num_ng - len(items)):
        #     neg_item = np.random.randint(self.num_item)
        #     neg_item = get_input_tensor(
        #         self.graph,
        #         neg_item + 2,
        #         self.hop_sampling_sizes,
        #         self.max_num_ctx_neigh,
        #     )
        #     while (user, neg_item) in self.user_item_mat:
        #         neg_item = np.random.randint(self.num_item)
        #     candidate_items.append(neg_item)
        #     labels.append(0)

        # return ((user, candidate_items), torch.FloatTensor(labels))


def pmgt_ncf_collate_fn(
    batch: Iterable[Tuple[torch.Tensor, ...]]
) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor], torch.FloatTensor]:
    user = torch.LongTensor([b[0][0] for b in batch])
    print(batch[0][0])
    if isinstance(batch[0][0][1], tuple):
        item = {
            "node_ids": torch.stack([b[0][1][0] for b in batch]),
            "attention_mask": torch.stack([b[0][1][1] for b in batch]),
        }
        label = torch.FloatTensor([b[1] for b in batch])
    else:
        item = torch.stack([b[0][1] for b in batch])
        label = None

    return (user, item), label
