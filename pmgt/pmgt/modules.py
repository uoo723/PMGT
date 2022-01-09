"""
Created on 2022/01/08
@author Sangwoo Han
@ref https://github.com/jwzhanggy/Graph-Bert/blob/master/code/MethodBertComp.py
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertPredictionHeadTransform,
)


class PMGTConfig(PretrainedConfig):

    model_type = "pmgt"

    def __init__(
        self,
        node_size=7252,
        hidden_size=128,
        feat_sizes=[1536, 768],
        num_hidden_layers=5,
        num_attention_heads=1,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.node_size = node_size
        self.hidden_size = hidden_size
        self.feat_sizes = feat_sizes
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class PMGTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [PMGTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PMGTEmbeddings(nn.Module):
    """Construct the embeddings from features, wl, position and hop vectors."""

    # def __init__(self, config):
    #     super().__init__()
    #     self.raw_feature_embeddings = nn.Linear(config.x_size, config.hidden_size)
    #     self.wl_role_embeddings = nn.Embedding(
    #         config.max_wl_role_index, config.hidden_size
    #     )
    #     self.inti_pos_embeddings = nn.Embedding(
    #         config.max_inti_pos_index, config.hidden_size
    #     )
    #     self.hop_dis_embeddings = nn.Embedding(
    #         config.max_hop_dis_index, config.hidden_size
    #     )

    #     self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # def forward(
    #     self, raw_features=None, wl_role_ids=None, init_pos_ids=None, hop_dis_ids=None
    # ):

    #     raw_feature_embeds = self.raw_feature_embeddings(raw_features)
    #     role_embeddings = self.wl_role_embeddings(wl_role_ids)
    #     position_embeddings = self.inti_pos_embeddings(init_pos_ids)
    #     hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)

    #     # ---- here, we use summation ----
    #     embeddings = (
    #         raw_feature_embeds + role_embeddings + position_embeddings + hop_embeddings
    #     )
    #     embeddings = self.LayerNorm(embeddings)
    #     embeddings = self.dropout(embeddings)
    #     return embeddings


class NodeConstructOutputLayer(nn.Module):
    pass
    # def __init__(self, config):
    #     super().__init__()
    #     self.transform = BertPredictionHeadTransform(config)

    #     # The output weights are the same as the input embeddings, but there is
    #     # an output-only bias for each token.
    #     self.decoder = nn.Linear(config.hidden_size, config.x_size, bias=False)

    #     self.bias = nn.Parameter(torch.zeros(config.x_size))

    #     # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
    #     self.decoder.bias = self.bias

    # def forward(self, hidden_states):
    #     hidden_states = self.transform(hidden_states)
    #     hidden_states = self.decoder(hidden_states) + self.bias
    #     return hidden_states


class PMGTLayer(nn.Module):
    pass
    # def __init__(self, config):
    #     super().__init__()
    #     self.chunk_size_feed_forward = config.chunk_size_feed_forward
    #     self.seq_len_dim = 1
    #     self.attention = BertAttention(config)
    #     self.intermediate = BertIntermediate(config)
    #     self.output = BertOutput(config)

    # def forward(
    #     self,
    #     hidden_states,
    #     attention_mask=None,
    #     head_mask=None,
    #     encoder_hidden_states=None,
    #     encoder_attention_mask=None,
    #     past_key_value=None,
    #     output_attentions=False,
    # ):
    #     self_attention_outputs = self.attention(
    #         hidden_states, attention_mask, head_mask
    #     )
    #     attention_output = self_attention_outputs[0]
    #     outputs = self_attention_outputs[
    #         1:
    #     ]  # add self attentions if we output attention weights

    #     if self.is_decoder and encoder_hidden_states is not None:
    #         cross_attention_outputs = self.crossattention(
    #             attention_output,
    #             attention_mask,
    #             head_mask,
    #             encoder_hidden_states,
    #             encoder_attention_mask,
    #         )
    #         attention_output = cross_attention_outputs[0]
    #         outputs = (
    #             outputs + cross_attention_outputs[1:]
    #         )  # add cross attentions if we output attention weights

    #     intermediate_output = self.intermediate(attention_output)
    #     layer_output = self.output(intermediate_output, attention_output)
    #     outputs = (layer_output,) + outputs
    #     return outputs
