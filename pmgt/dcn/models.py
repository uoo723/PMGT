"""
Created on 2022/01/06
@author Sangwoo Han
@ref https://github.com/brightnesss/deep-cross/blob/master/CDNet.py
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_hidden_size, output_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = (
            nn.LayerNorm(output_hidden_size, layer_norm_eps)
            if use_layer_norm
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.linear(inputs)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.act(outputs)
        return outputs


class CrossLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = (
            nn.LayerNorm(hidden_size, layer_norm_eps)
            if use_layer_norm
            else nn.Identity()
        )

        self._init_weights()

    def forward(
        self, inputs: Tuple[torch.FloatTensor, torch.FloatTensor]
    ) -> torch.FloatTensor:
        x0, x1 = inputs
        outputs = x0.unsqueeze(2) @ x1.unsqueeze(1)
        outputs = outputs @ self.weight
        outputs = outputs.squeeze()
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + x0)
        return x0, outputs

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)


class DeepNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        linear_size: List[int],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()

        linear_size = [input_size] + linear_size

        self.layers = nn.Sequential(
            *[
                MLPLayer(in_size, out_size, dropout, use_layer_norm, layer_norm_eps)
                for in_size, out_size in zip(linear_size[:-1], linear_size[1:])
            ]
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.layers(x)
        return outputs


class CrossNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[
                CrossLayer(input_size, dropout, use_layer_norm, layer_norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.layers((inputs, inputs))[1]
        return outputs


class DCN(nn.Module):
    """
    Cross and Deep Network in Deep & Cross Network for Ad Click Predictions
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        factor_num: int = 32,
        deep_net_num_layers: int = 3,
        cross_net_num_layers: int = 2,
        emb_dropout: float = 0.0,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        input_size = factor_num * (2 ** deep_net_num_layers)
        self.user_embeddings = nn.Embedding(user_num, input_size)
        self.item_embeddings = nn.Embedding(item_num, input_size)
        self.emb_dropout = nn.Dropout(emb_dropout)

        linear_size = [
            factor_num * (2 ** (deep_net_num_layers + 1 - i))
            for i in range(deep_net_num_layers + 1)
        ]

        self.deep_net = DeepNet(
            linear_size[0], linear_size[1:], dropout, use_layer_norm, layer_norm_eps
        )

        self.cross_net = CrossNet(
            input_size * 2,
            cross_net_num_layers,
            dropout,
            use_layer_norm,
            layer_norm_eps,
        )

        self.output_layer = nn.Linear(input_size * 2 + linear_size[-1], 1)

    def forward(
        self, inputs: Tuple[torch.LongTensor, torch.LongTensor]
    ) -> torch.FloatTensor:
        user, item = inputs
        user_embed = self.user_embeddings(user)
        item_embed = self.item_embeddings(item)

        interaction = torch.cat([user_embed, item_embed], dim=-1)
        interaction = self.emb_dropout(interaction)

        cross_net_outputs = self.cross_net(interaction)
        deep_net_outputs = self.deep_net(interaction)

        outputs = torch.cat([cross_net_outputs, deep_net_outputs], dim=-1)
        outputs: torch.FloatTensor = self.output_layer(outputs)

        return outputs.view(-1)
