"""
Created on 2022/01/08
@author Sangwoo Han
"""
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.special as ss
import torch


class PMGTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graph: nx.Graph,
        node_ids: Optional[np.ndarray] = None,
        max_ctx_neigh: int = 20,
        sampling_sizes: List[int] = [16, 8, 4],
        num_ng: int = 0,
        is_training: bool = True,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.node_ids = node_ids if node_ids is not None else np.arange(len(self.graph))
        self.max_num_ctx_neigh = max_ctx_neigh
        self.sampling_sizes = sampling_sizes
        self.is_training = is_training

        self.depth = len(self.sampling_sizes)

    def __len__(self) -> int:
        return len(self.node_ids)  # num of nodes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx_nodes, num_ctx_nodes = self._sample_context_neigh(self.node_ids[idx] + 1)
        assert (
            len(ctx_nodes) == self.max_num_ctx_neigh
        ), f"# of context nodes must be {self.max_num_ctx_neigh}"

        attention_mask = torch.zeros(self.max_num_ctx_neigh + 1, dtype=torch.float32)
        attention_mask[: num_ctx_nodes + 1] = 1

        return torch.LongTensor([self.node_ids[idx] + 1] + ctx_nodes), attention_mask

    def _sample_context_neigh(self, target_node: int) -> Tuple[List, int]:
        scores = defaultdict(lambda: 0)

        sampled_nodes = [[target_node]] + [[] for _ in range(self.depth)]

        for k, sample_size in enumerate(self.sampling_sizes, start=1):
            for node in sampled_nodes[k - 1]:
                weights = ss.softmax(
                    np.asarray([v["weight"] for v in self.graph[node].values()])
                )
                node_list = np.random.choice(
                    self.graph[node], size=sample_size, replace=True, p=weights
                ).tolist()
                sampled_nodes[k].extend(node_list)

            counter = Counter(sampled_nodes[k])

            for node, freq in counter.items():
                if node == target_node:
                    continue
                scores[node] += freq * (self.depth - k + 1)

        ctx_nodes, _ = zip(
            *sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )

        ctx_nodes = list(ctx_nodes)

        if len(ctx_nodes) < self.max_num_ctx_neigh:
            num_ctx_nodes = len(ctx_nodes)
            ctx_nodes += [0] * (self.max_num_ctx_neigh - len(ctx_nodes))
        else:
            num_ctx_nodes = self.max_num_ctx_neigh
            ctx_nodes = ctx_nodes[: self.max_num_ctx_neigh]

        return ctx_nodes, num_ctx_nodes


def pmgt_collate_fn(
    batch: Iterable[Tuple[torch.Tensor, ...]]
) -> Dict[str, torch.Tensor]:
    inputs = {
        "node_ids": torch.stack([b[0] for b in batch]),
        "attention_mask": torch.stack([b[1] for b in batch]),
    }
    return inputs
