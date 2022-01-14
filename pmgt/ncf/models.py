"""
Created on 2022/01/06
@author Sangwoo Han
@ref https://github.com/guoyang9/NCF/blob/master/model.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class NCF(nn.Module):
    """Implementation of "Neural Collaborative Filtering"

    Args:
        user_num (int): number of users
        item_num (int): number of items
        factor_num (int): number of predictive factors
        num_layers (int): the number of layers in MLP model
        dropout (float): dropout rate between fully connected layers
        model (str): 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre'
        GMF_model (NCF): pre-trained GMF weights
        MLP_model (NCF): pre-trained MLP weights
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        factor_num: int = 32,
        num_layers: int = 3,
        emb_dropout: float = 0.0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
        model: str = "NeuMF-end",
        GMF_model: Optional[NCF] = None,
        MLP_model: Optional[NCF] = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        assert model in ["MLP", "GMF", "NeuMF-end", "NeuMF-pre"]

        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.alpha = alpha

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.LayerNorm(input_size // 2, eps=layer_norm_eps))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ["MLP", "GMF"]:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight()

    def _init_weight(self) -> None:
        """We leave the weights initialization here."""
        if not self.model == "NeuMF-pre":
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(
                self.predict_layer.weight, a=1, nonlinearity="sigmoid"
            )

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat(
                [
                    self.alpha * self.GMF_model.predict_layer.weight,
                    (1 - self.alpha) * self.MLP_model.predict_layer.weight,
                ],
                dim=1,
            )
            precit_bias = (
                self.alpha * self.GMF_model.predict_layer.bias
                + (1 - self.alpha) * self.MLP_model.predict_layer.bias
            )

            self.predict_layer.weight.data.copy_(predict_weight)
            self.predict_layer.bias.data.copy_(precit_bias)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        user, item = inputs
        if not self.model == "MLP":
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = self.emb_dropout(embed_user_GMF * embed_item_GMF)
        if not self.model == "GMF":
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = self.emb_dropout(
                torch.cat((embed_user_MLP, embed_item_MLP), -1)
            )
            output_MLP = self.MLP_layers(interaction)

        if self.model == "GMF":
            concat = output_GMF
        elif self.model == "MLP":
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
