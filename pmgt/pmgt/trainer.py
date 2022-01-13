"""
Created on 2022/01/08
@author Sangwoo Han
"""
import os
from typing import Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from attrdict import AttrDict
from logzero import logger
from optuna import Trial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from pmgt.pmgt.configuration_pmgt import PMGTConfig

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from .datasets import PMGTDataset, pmgt_collate_fn
from .models import PMGT


def _get_dataset(args: AttrDict) -> Tuple[PMGTDataset, PMGTDataset, PMGTDataset]:
    if args.run_id is not None:
        _set_dataset_param(args)

    data_dir = os.path.join(args.data_dir, args.dataset_name)
    node_encoder: LabelEncoder = joblib.load(os.path.join(data_dir, "node_encoder"))
    graph = nx.read_gpickle(os.path.join(data_dir, "graph.gpickle"))

    # idx 0 is <pad>
    # idx 1 is <mask>
    mapping = {label: i + 2 for i, label in enumerate(node_encoder.classes_)}
    graph = nx.relabel_nodes(graph, mapping)

    args.graph = graph

    train_nodes, valid_nodes = train_test_split(
        np.arange(
            start=2,
            stop=len(graph) + 2,
        ),
        test_size=args.valid_size,
        random_state=args.seed,
    )

    train_dataset = PMGTDataset(
        graph,
        train_nodes,
        args.max_ctx_neigh,
        args.hop_sampling_sizes,
        args.max_total_samples,
        args.min_neg_samples,
    )

    valid_dataset = PMGTDataset(
        graph,
        valid_nodes,
        args.max_ctx_neigh,
        args.hop_sampling_sizes,
        is_training=False,
    )

    return train_dataset, valid_dataset, valid_dataset


def _set_dataset_param(args: AttrDict) -> None:
    run = base_trainer.get_run(args.log_dir, args.run_id)
    params = AttrDict(run.data.params)

    args.max_ctx_neigh = int(params.max_ctx_neigh)
    args.max_total_samples = int(params.max_total_samples)
    args.min_neg_samples = int(params.min_neg_samples)
    args.hop_sampling_sizes = eval(params.hop_sampling_sizes)


def _get_dataloader(
    args: AttrDict,
    train_dataset: PMGTDataset,
    valid_dataset: PMGTDataset,
    test_dataset: PMGTDataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pmgt_collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        collate_fn=pmgt_collate_fn,
    )

    return train_dataloader, valid_dataloader, valid_dataloader


def _get_model(args: AttrDict) -> nn.Module:
    if args.run_id is not None:
        _set_model_param(args)

    data_dir = os.path.join(args.data_dir, args.dataset_name)

    visual_init_emb = np.load(os.path.join(data_dir, "visual_init_emb.npy"))
    textual_init_emb = np.load(os.path.join(data_dir, "textual_init_emb.npy"))
    feat_hidden_sizes = [visual_init_emb.shape[-1], textual_init_emb.shape[-1]]

    config = PMGTConfig(
        hidden_size=args.hidden_size,
        feat_hidden_sizes=feat_hidden_sizes,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        beta=args.beta,
    )

    model = PMGT(
        node_size=len(args.graph),
        random_node_ratio=args.random_node_ratio,
        mask_node_ratio=args.mask_node_ratio,
        config=config,
        feat_init_emb=[visual_init_emb, textual_init_emb],
    )

    return model


def _set_model_param(args: AttrDict) -> None:
    run = base_trainer.get_run(args.log_dir, args.run_id)
    params = AttrDict(run.data.params)
    args.hidden_size = int(params.hidden_size)
    args.intermediate_size = int(params.intermediate_size)
    args.num_hidden_layers = int(params.num_hidden_layers)
    args.num_attention_heads = int(params.num_attention_heads)
    args.beta = float(params.beta)
    args.random_node_ratio = float(params.random_node_ratio)
    args.mask_node_ratio = float(params.mask_node_ratio)


class PMGTTrainerModel(BaseTrainerModel):
    IGNORE_HPARAMS: List[str] = BaseTrainerModel.IGNORE_HPARAMS + ["graph"]

    def forward(self, x):
        return self.net(x)[0][:, 0].cpu().numpy()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.net(*batch)[0]
        batch_size = self.args.train_dataloader.batch_size
        self.log("loss/train", loss, batch_size=batch_size)
        return loss

    def _validation_and_test_step(
        self, batch, log_loss_name: Optional[str] = None
    ) -> np.ndarray:
        outputs = self.net(*batch)
        loss: torch.FloatTensor = outputs[0]
        logits: torch.FloatTensor = outputs[1]
        labels: torch.FloatTensor = batch[-1]

        if log_loss_name:
            batch_size = self.args.valid_dataloader.batch_size
            self.log(log_loss_name, loss, batch_size=batch_size)

        return logits.sigmoid().cpu().numpy(), labels.cpu().numpy()

    def validation_step(self, batch, batch_idx: int) -> np.ndarray:
        return self._validation_and_test_step(batch, "loss/val")

    def test_step(self, batch, batch_idx: int) -> np.ndarray:
        return self._validation_and_test_step(batch)

    def _valid_and_test_epoch_end(
        self, outputs: Tuple[List[np.ndarray], List[np.ndarray]], log_auc_name: str
    ) -> Dict[str, float]:
        batch_size = self.args.valid_dataloader.batch_size

        predictions, labels = zip(*outputs)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        auc = roc_auc_score(labels, predictions)
        results = {log_auc_name: auc}
        self.log_dict(results, prog_bar=True, batch_size=batch_size)

        return results

    def validation_epoch_end(
        self, outputs: Tuple[List[np.ndarray], List[np.ndarray]]
    ) -> None:
        results = self._valid_and_test_epoch_end(outputs, "val/auc")
        self.should_prune(results["val/auc"])

    def test_epoch_end(
        self, outputs: Tuple[List[np.ndarray], List[np.ndarray]]
    ) -> None:
        self._valid_and_test_epoch_end(outputs, "test/auc")


def init_run(*args, **kwargs):
    base_trainer.init_run(*args, **kwargs)


def check_args(args: AttrDict) -> None:
    early_criterion = ["loss", "auc"]
    model_name = ["PMGT"]
    dataset_name = ["VG"]

    base_trainer.check_args(args, early_criterion, model_name, dataset_name)


def init_dataloader(args: AttrDict) -> None:
    base_trainer.init_dataloader(args, _get_dataset, _get_dataloader)

    logger.info(f"# of train dataset: {len(args.train_dataset):,}")
    logger.info(f"# of valid dataset: {len(args.valid_dataset):,}")


def init_model(args: AttrDict) -> None:
    base_trainer.init_model(args, _get_model)


def train(
    args: AttrDict,
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
) -> Tuple[float, pl.Trainer]:
    return base_trainer.train(
        args,
        PMGTTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        PMGTTrainerModel,
        metrics=["auc"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def inference(args: AttrDict) -> np.ndarray:
    dataset = PMGTDataset(
        args.graph,
        max_ctx_neigh=args.max_ctx_neigh,
        hop_sampling_sizes=args.hop_sampling_sizes,
        is_training=False,
        is_inference=True,
    )

    args.inference_dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        collate_fn=pmgt_collate_fn,
    )

    return base_trainer.inference(args, PMGTTrainerModel)
