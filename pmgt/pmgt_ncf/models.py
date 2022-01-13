"""
Created on 2022/01/13
@author Sangwoo Han
"""
from typing import Dict

import torch
import torch.nn as nn

from ..pmgt.configuration_pmgt import PMGTConfig
from ..pmgt.modeling_pmgt import PMGTModel, PMGTPretrainedModel
from ..pmgt.utils import get_input_feat_embeds


class PMGT_NCF(PMGTPretrainedModel):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        factor_num: int = 32,
        num_layers: int = 3,
        emb_dropout: float = 0.0,
        dropout: float = 0.0,
        config: PMGTConfig = PMGTConfig(),
    ) -> None:
        super().__init__(config)

        self.factor_num = factor_num
        self.num_layers = num_layers

        self.bert = PMGTModel(config)
        self.feat_embeddings = nn.ModuleList(
            [
                # idx 0 is <pad>
                # idx 1 is <mask>
                nn.Embedding(item_num + 2, feat_hidden_size, padding_idx=0)
                for feat_hidden_size in config.feat_hidden_sizes
            ]
        )

        # Freeze multi-modal feature embeddings
        for feat_embeddings in self.feat_embeddings:
            feat_embeddings.requires_grad_(False)

        # NCF Part
        self.user_embeddings = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.mlp_layers = nn.Sequential(
            *[
                MLPLayer(
                    self._get_input_size(i),
                    self._get_input_size(i) // 2,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.predict_layer = nn.Linear(factor_num, 1)

    def _get_input_size(self, i: int) -> int:
        return self.factor_num * 2 ** (self.num_layers - i)

    def forward(
        self, user: torch.LongTensor, item: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        user_embeds = self.user_embeddings(user)
        input_feat_embeds = get_input_feat_embeds(
            item["node_ids"], self.feat_embeddings
        )
        item_embeds = self.bert(
            *input_feat_embeds,
            attention_mask=item["attention_mask"],
        )[0][:, 0]

        interaction = torch.cat([user_embeds, item_embeds], dim=-1)
        interaction = self.emb_dropout(interaction)
        outputs = self.mlp_layers(interaction)

        logits: torch.FloatTensor = self.predict_layer(outputs)
        return logits.view(-1)


class MLPLayer(nn.Module):
    def __init__(
        self, input_hidden_size: int, output_hidden_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_hidden_size, output_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.act(hidden_states)
        return hidden_states
