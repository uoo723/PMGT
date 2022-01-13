"""
Created on 2022/01/08
@author Sangwoo Han
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .configuration_pmgt import PMGTConfig
from .modeling_pmgt import (
    PMGTForPreTrainingOutput,
    PMGTGraphConstructLoss,
    PMGTModel,
    PMGTNodeConstructLoss,
    PMGTPretrainedModel,
)
from .utils import get_input_feat_embeds


class PMGT(PMGTPretrainedModel):
    def __init__(
        self,
        node_size: int,
        random_node_ratio: float = 0.2 * 0.1,
        mask_node_ratio: float = 0.2 * 0.8,
        config: PMGTConfig = PMGTConfig(),
        feat_init_emb: Optional[List[np.ndarray]] = None,
    ) -> None:
        super().__init__(config)
        self.node_size = node_size
        self.random_node_ratio = random_node_ratio
        self.mask_node_ratio = mask_node_ratio
        self.config = config
        self.bert = PMGTModel(config)
        self.gsr_loss = PMGTGraphConstructLoss(config)
        self.nfr_loss = PMGTNodeConstructLoss(config)

        self.feat_embeddings = nn.ModuleList(
            [
                # idx 0 is <pad>
                # idx 1 is <mask>
                nn.Embedding(node_size + 2, feat_hidden_size, padding_idx=0)
                for feat_hidden_size in config.feat_hidden_sizes
            ]
        )

        if feat_init_emb is not None:
            assert len(feat_init_emb) == len(self.feat_embeddings)
            for feat_embeddings, weights in zip(self.feat_embeddings, feat_init_emb):
                with torch.no_grad():
                    feat_embeddings.weight.copy_(torch.from_numpy(weights))
                feat_embeddings.requires_grad_(False)

    def forward(
        self,
        target_node_inputs: Dict[str, torch.Tensor],
        pair_node_inputs: Optional[Dict[str, torch.Tensor]] = None,
        num_pairs: torch.LongTensor = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        if pair_node_inputs is not None:
            assert (
                labels is not None
            ), "labels must be passed, when set pair_node_inputs"
            assert (
                num_pairs is not None
            ), "num_pairs must be passed, when set pair_node_inputs"

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        device = target_node_inputs["node_ids"].device
        input_feat_embeds = get_input_feat_embeds(
            target_node_inputs["node_ids"], self.feat_embeddings
        )

        bert_outputs = self.bert(
            *input_feat_embeds,
            attention_mask=target_node_inputs["node_ids"],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        prediction_logits = None

        if pair_node_inputs is not None:
            ############################## GSR Loss ###################################
            gsr_loss = []
            prediction_logits = []
            target_outputs = bert_outputs[0]

            bs = 0
            for i, num in enumerate(num_pairs):
                be = bs + num.item()
                pair_outputs = self.bert(
                    *get_input_feat_embeds(
                        pair_node_inputs["node_ids"][bs:be], self.feat_embeddings
                    ),
                    attention_mask=pair_node_inputs["attention_mask"][bs:be],
                )[0]
                loss, logits = self.gsr_loss(
                    pair_outputs[:, 0], target_outputs[i, 0], labels[bs:be]
                )
                gsr_loss.append(loss)
                prediction_logits.append(logits)
                bs = be
            gsr_loss = torch.stack(gsr_loss).mean()
            prediction_logits = torch.cat(prediction_logits)
            ###########################################################################

            nfr_loss = 0
            if self.training:
                ############################### NFR Loss ##################################
                masked_input_ids = target_node_inputs["node_ids"].clone()
                input_shape = masked_input_ids.size()
                # except target node
                rand = torch.rand(input_shape[0], input_shape[1] - 1, device=device)
                # Random node replacement
                mask = (rand < self.random_node_ratio) * (masked_input_ids[:, 1:] != 0)
                masked_input_ids[:, 1:][mask] = torch.randint(
                    2,
                    self.node_size + 2,
                    (mask.sum(),),
                    device=device,
                )

                # Masking node
                # except target node
                rand = torch.rand(input_shape[0], input_shape[1] - 1, device=device)
                # except padding
                mask = (rand < self.mask_node_ratio) * (masked_input_ids[:, 1:] != 0)
                target_idx = masked_input_ids[:, 1:][mask]
                masked_input_ids[:, 1:][mask] = 1  # Fill mask index

                masked_outputs = self.bert(
                    *get_input_feat_embeds(masked_input_ids, self.feat_embeddings),
                    attention_mask=target_node_inputs["attention_mask"],
                )[0]

                masked_outputs = masked_outputs[:, 1:][mask]
                target_embeds = get_input_feat_embeds(target_idx, self.feat_embeddings)

                nfr_loss = self.nfr_loss(masked_outputs, target_embeds)
                ###########################################################################

            loss = gsr_loss + nfr_loss

        if not return_dict:
            return (loss, prediction_logits) + bert_outputs

        return PMGTForPreTrainingOutput(
            loss=loss,
            prediction_logits=prediction_logits,
            last_hidden_state=bert_outputs.last_hidden_state,
            pooler_output=bert_outputs.pooler_output,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )
