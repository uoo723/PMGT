"""
Created on 2022/01/09
@author Sangwoo Han
@ref https://github.com/jwzhanggy/Graph-Bert/blob/master/code/MethodBertComp.py
"""
from transformers.configuration_utils import PretrainedConfig


class PMGTConfig(PretrainedConfig):

    model_type = "pmgt"

    def __init__(
        self,
        hidden_size=128,
        feat_hidden_sizes=[1536, 768],
        num_hidden_layers=5,
        num_attention_heads=1,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        beta=0.5,  # diversity promoting attention weight
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.feat_hidden_sizes = feat_hidden_sizes
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.beta = beta
