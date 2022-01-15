"""
Created on 2022/01/15
@author Sangwoo Han
"""
import os
from distutils.util import strtobool
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from attrdict import AttrDict
from logzero import logger
from optuna import Trial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..pmgt.utils import load_node_init_emb
from .datasets import DCNDataset
from .models import DCN

TInput = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor]]
TOutput = torch.Tensor
TBatch = Tuple[TInput, TOutput]


def _get_dataset(args: AttrDict) -> Tuple[DCNDataset, DCNDataset, DCNDataset]:
    data_dir = os.path.join(args.data_dir, args.dataset_name)
    user_encoder = joblib.load(os.path.join(data_dir, "user_encoder"))
    item_encoder = joblib.load(os.path.join(data_dir, "item_encoder"))
    train_df = pd.read_json(os.path.join(data_dir, "train.json"))
    test_df = pd.read_json(os.path.join(data_dir, "test.json"))

    train_user_ids = user_encoder.transform(train_df["reviewerID"].values)
    train_item_ids = item_encoder.transform(train_df["asin"].values)
    train_data = list(zip(train_user_ids, train_item_ids))

    test_user_ids = user_encoder.transform(test_df["reviewerID"].values)
    test_item_ids = item_encoder.transform(test_df["asin"].values)
    test_data = list(zip(test_user_ids, test_item_ids))

    train_data, valid_data = train_test_split(
        train_data, test_size=args.valid_size, random_state=args.seed
    )

    args.num_user = len(user_encoder.classes_)
    args.num_item = len(item_encoder.classes_)

    train_dataset = DCNDataset(
        train_data,
        args.num_user,
        args.num_item,
        num_ng=args.num_ng,
    )
    valid_dataset = DCNDataset(
        valid_data,
        args.num_user,
        args.num_item,
        num_ng=args.max_sample_items,
        # is_training=False,
    )
    test_dataset = DCNDataset(
        test_data,
        args.num_user,
        args.num_item,
        num_ng=args.max_sample_items,
        # is_training=False,
    )

    train_dataset.ng_sample()
    valid_dataset.ng_sample()
    test_dataset.ng_sample()

    return train_dataset, valid_dataset, test_dataset


def _get_dataloader(
    args: AttrDict,
    train_dataset: DCNDataset,
    valid_dataset: DCNDataset,
    test_dataset: DCNDataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def _get_model(args: AttrDict) -> nn.Module:
    if args.run_id is not None:
        _set_model_param(args)

    model = DCN(
        args.num_user,
        args.num_item,
        args.factor_num,
        args.deep_net_num_layers,
        args.cross_net_num_layers,
        args.emb_dropout,
        args.dropout,
        args.use_layer_norm,
        args.layer_norm_eps,
    )

    if args.item_init_emb_path is not None:
        data_dir = os.path.join(args.data_dir, args.dataset_name)
        item_encoder_path = os.path.join(data_dir, "item_encoder")
        node_encoder_path = os.path.join(data_dir, "node_encoder")
        item_init_emb = load_node_init_emb(
            item_encoder_path,
            node_encoder_path,
            args.item_init_emb_path,
            args.normalize_item_init_emb,
        )
        model.item_embeddings.weight.data.copy_(torch.from_numpy(item_init_emb))
        model.item_embeddings.requires_grad_(not args.freeze_item_init_emb)
    return model


def _set_model_param(args: AttrDict) -> None:
    run = base_trainer.get_run(args.log_dir, args.run_id)
    params = AttrDict(run.data.params)
    args.factor_num = int(params.factor_num)
    args.deep_net_num_layers = int(params.deep_net_num_layers)
    args.cross_net_num_layers = int(params.cross_net_num_layers)
    args.emb_dropout = float(params.emb_dropout)
    args.dropout = float(params.dropout)
    args.use_layer_norm = bool(strtobool(params.use_layer_norm))
    args.layer_norm_eps = float(params.layer_norm_eps)


class DCNTrainerModel(BaseTrainerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x: TInput) -> TOutput:
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        if self.global_step > 0:
            self.args.train_dataset.ng_sample()

    def training_step(self, batch: TBatch, batch_idx: int) -> torch.Tensor:
        batch_x, batch_y = batch
        pred = self.net(batch_x)
        loss = self.loss_func(pred, batch_y)
        self.log("loss/train", loss)
        return loss

    def _validation_and_test_step(
        self, batch: TBatch, log_loss_name: Optional[str] = None
    ) -> np.ndarray:
        batch_x, batch_y = batch
        pred: torch.FloatTensor = self.net(batch_x)
        loss = self.loss_func(pred, batch_y)

        if log_loss_name:
            self.log(log_loss_name, loss)

        return pred.sigmoid().cpu().numpy()

    def validation_step(self, batch: TBatch, batch_idx: int) -> np.ndarray:
        return self._validation_and_test_step(batch, "loss/val")

    def test_step(self, batch: TBatch, batch_idx: int) -> np.ndarray:
        return self._validation_and_test_step(batch)

    def _valid_and_test_epoch_end(
        self, outputs: List[np.ndarray], log_auc_name: str, is_test: bool = False
    ) -> Dict[str, float]:
        dataset = (
            self.args.valid_dataset
            if not is_test or self.is_hptuning
            else self.args.test_dataset
        )

        predictions = np.concatenate(outputs)
        gt = dataset.gt

        auc = (
            roc_auc_score(gt, predictions)
            if gt.shape[0] == predictions.shape[0]
            else 0.0
        )

        results = {log_auc_name: auc}
        self.log_dict(results, prog_bar=True)

        return results

    def validation_epoch_end(self, outputs: List[np.ndarray]) -> None:
        results = self._valid_and_test_epoch_end(outputs, "val/auc")
        self.should_prune(results["val/auc"])

    def test_epoch_end(self, outputs: List[np.ndarray]) -> None:
        self._valid_and_test_epoch_end(outputs, "test/auc", is_test=True)


def init_run(*args, **kwargs):
    base_trainer.init_run(*args, **kwargs)


def check_args(args: AttrDict) -> None:
    early_criterion = ["loss", "auc"]
    model_name = ["DCN"]
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
        DCNTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        DCNTrainerModel,
        metrics=["auc"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def inference(args: AttrDict):
    raise NotImplemented
