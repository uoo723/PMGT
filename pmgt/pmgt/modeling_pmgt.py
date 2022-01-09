"""
Created on 2022/01/08
@author Sangwoo Han
@ref https://github.com/jwzhanggy/Graph-Bert/blob/master/code/MethodBertComp.py
"""

import math

import torch
import torch.linalg
import torch.nn as nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertOutput,
    BertSelfOutput,
)

from .configuration_pmgt import PMGTConfig


class PMGTPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PMGTConfig
    base_model_prefix = "pmgt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PMGTEncoder):
            module.gradient_checkpointing = value


class PMGTModel(PMGTPretrainedModel):
    pass
    # def __init__(self, config, add_pooling_layer=True):
    #     super().__init__(config)
    #     self.config = config

    #     self.embeddings = PMGTEmbeddings(config)
    #     self.encoder = PMGTEncoder(config)
    #     self.pooler = BertPooler(config) if add_pooling_layer else None

    #     self.init_weights()

    # def get_input_embeddings(self):
    #     return self.embeddings.raw_feature_embeddings

    # def set_input_embeddings(self, value):
    #     self.embeddings.raw_feature_embeddings = value

    # def _prune_heads(self, heads_to_prune):
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    # def forward(
    #     self,
    #     raw_features,
    #     wl_role_ids,
    #     init_pos_ids,
    #     hop_dis_ids,
    #     head_mask=None,
    #     residual_h=None,
    # ):
    #     if head_mask is None:
    #         head_mask = [None] * self.config.num_hidden_layers

    #     embedding_output = self.embeddings(
    #         raw_features=raw_features,
    #         wl_role_ids=wl_role_ids,
    #         init_pos_ids=init_pos_ids,
    #         hop_dis_ids=hop_dis_ids,
    #     )
    #     encoder_outputs = self.encoder(
    #         embedding_output, head_mask=head_mask, residual_h=residual_h
    #     )
    #     sequence_output = encoder_outputs[0]
    #     pooled_output = self.pooler(sequence_output)
    #     outputs = (
    #         sequence_output,
    #         pooled_output,
    #     ) + encoder_outputs[1:]
    #     return outputs


class PMGTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_embeddings = nn.ModuleList(
            nn.Embedding(config.node_size, feat_hidden_size)
            for feat_hidden_size in config.feat_hidden_sizes
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.role_embeddings = nn.Embedding(2, config.hidden_size)

        self.feat_linear = nn.ModuleList(
            nn.Linear(feat_hidden_size, config.hidden_size)
            for feat_hidden_size in config.feat_hidden_sizes
        )

        num_feats = len(config.feat_hidden_sizes)
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(num_feats * config.hidden_size, num_feats),
            nn.Softmax(dim=-1),
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "postition_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
        )
        self.register_buffer(
            "role_ids",
            torch.LongTensor(
                [0] + [1] * (config.max_position_embeddings - 1)
            ).unsqueeze(0),
        )

        if config.freeze_feat_embeddings:
            for feat_emb in self.feat_embeddings:
                feat_emb.requires_grad_(False)

    def forward(self, node_ids):
        seq_len = node_ids.size(1)

        position_ids = self.postition_ids[:, :seq_len]
        role_ids = self.role_ids[:, :seq_len]

        feat_embeds = [
            feat_linear(feat_embeddings(node_ids))
            for feat_embeddings, feat_linear in zip(
                self.feat_embeddings, self.feat_linear
            )
        ]

        attention_scores = self.attention(torch.cat(feat_embeds, dim=-1))
        feat_embeds = attention_scores.unsqueeze(-1) * torch.stack(feat_embeds, dim=2)
        feat_embeds = feat_embeds.sum(dim=2)

        position_embeds = self.position_embeddings(position_ids)
        role_embeds = self.role_embeddings(role_ids)

        embeds = feat_embeds + position_embeds + role_embeds
        embeds = self.LayerNorm(embeds)
        embeds = self.dropout(embeds)

        return embeds


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
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
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


class PMGTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PMGTAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class PMGTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = PMGTSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class PMGTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.beta = config.beta

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.ctx_attention = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        seq_len = hidden_states.size(1)

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        ctx_layer = self.transpose_for_scores(self.ctx_attention(hidden_states))

        ctx_norm = torch.linalg.norm(ctx_layer, dim=-1, keepdim=True)
        ctx_norm = torch.matmul(ctx_norm, ctx_norm.transpose(-1, -2))

        all_ones = torch.ones(
            1,
            1,
            seq_len,
            seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        identity = torch.eye(
            seq_len, device=hidden_states.device, dtype=hidden_states.dtype
        ).expand(1, 1, seq_len, seq_len)

        attention_scores1 = torch.matmul(ctx_layer, ctx_layer.transpose(-1, -2))
        attention_scores1 = attention_scores1 / ctx_norm
        attention_scores1 = all_ones - attention_scores1 + identity

        if attention_mask is not None:
            attention_scores1 = attention_scores1 + attention_mask

        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs1 = self.dropout(attention_probs1)

        if head_mask is not None:
            attention_probs1 = attention_probs1 * head_mask

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores2 = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores2 = attention_scores2 + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores2 = (
                    attention_scores2
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores2 = attention_scores2 + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout(attention_probs2)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs2 = attention_probs2 * head_mask

        weighted_attention_probs = (
            self.beta * attention_probs1 + (1 - self.beta) * attention_probs2
        )

        context_layer = torch.matmul(weighted_attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, weighted_attention_probs)
            if output_attentions
            else (context_layer,)
        )

        return outputs


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
