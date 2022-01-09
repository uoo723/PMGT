"""
Created on 2022/01/08
@author Sangwoo Han
@ref https://github.com/jwzhanggy/Graph-Bert/blob/master/code/MethodGraphBert.py
"""

import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertPooler, BertPreTrainedModel

from .configuration_pmgt import PMGTConfig
from .modeling_pmgt import PMGTEmbeddings, PMGTEncoder


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


class PMGT(PMGTPretrainedModel):
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
