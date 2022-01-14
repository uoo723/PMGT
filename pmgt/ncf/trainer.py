"""
Created on 2022/01/08
@author Sangwoo Han
"""
import os
from collections import OrderedDict
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .. import base_trainer
from ..base_trainer import BaseTrainerModel, get_ckpt_path, get_run
from ..metrics import get_ndcg, get_recall
from ..pmgt.utils import load_node_init_emb
from .datasets import NCFDataset
from .models import NCF

TInput = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor]]
TOutput = torch.Tensor
TBatch = Tuple[TInput, TOutput]


def _get_dataset(args: AttrDict) -> Tuple[NCFDataset, NCFDataset, NCFDataset]:
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

    train_dataset = NCFDataset(
        train_data, args.num_user, args.num_item, num_ng=args.num_ng
    )
    valid_dataset = NCFDataset(
        valid_data,
        args.num_user,
        args.num_item,
        num_ng=args.max_sample_items,
        is_training=False,
    )
    test_dataset = NCFDataset(
        test_data,
        args.num_user,
        args.num_item,
        num_ng=args.max_sample_items,
        is_training=False,
    )

    train_dataset.ng_sample()

    return train_dataset, valid_dataset, test_dataset


def _get_dataloader(
    args: AttrDict,
    train_dataset: NCFDataset,
    valid_dataset: NCFDataset,
    test_dataset: NCFDataset,
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
        num_workers=args.num_workers * 2,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers * 2,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def _load_pretrained_model(
    log_dir: str,
    run_id: str,
    GMF_model: Optional[NCF] = None,
    MLP_model: Optional[NCF] = None,
) -> NCF:
    ckpt_path = get_ckpt_path(log_dir, run_id, load_best=True)
    params = AttrDict(get_run(log_dir, run_id).data.params)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = NCF(
        user_num=int(params.num_user),
        item_num=int(params.num_item),
        factor_num=int(params.factor_num),
        num_layers=int(params.num_layers),
        emb_dropout=float(getattr(params, "emb_dropout", 0.0)),
        dropout=float(params.dropout),
        use_layer_norm=bool(strtobool(getattr(params, "use_layer_norm", "False"))),
        layer_norm_eps=float(getattr(params, "layer_norm_eps", 1e-12)),
        model=params.model_name,
        GMF_model=GMF_model,
        MLP_model=MLP_model,
    )

    model.load_state_dict(
        OrderedDict([(k.replace("net.", ""), v) for k, v in ckpt["state_dict"].items()])
    )

    return model


def _get_model(args: AttrDict) -> nn.Module:
    if args.model_name == "NeuMF-pre":
        GMF_model = _load_pretrained_model(args.log_dir, args.gmf_run_id)
        MLP_model = _load_pretrained_model(args.log_dir, args.mlp_run_id)
    else:
        GMF_model = MLP_model = None

    if args.run_id is not None:
        model = _load_pretrained_model(args.log_dir, args.run_id, GMF_model, MLP_model)
    else:
        model = NCF(
            args.num_user,
            args.num_item,
            args.factor_num,
            args.num_layers,
            args.emb_dropout,
            args.dropout,
            args.use_layer_norm,
            args.layer_norm_eps,
            args.model_name,
            GMF_model=GMF_model,
            MLP_model=MLP_model,
            alpha=args.alpha,
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
        model.embed_item_MLP.weight.data.copy_(torch.from_numpy(item_init_emb))
        model.embed_item_MLP.requires_grad_(not args.freeze_item_init_emb)
    return model


class NCFTrainerModel(BaseTrainerModel):
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
        users, items = batch_x
        users = users.unsqueeze(1).repeat(items.size())
        predictions = []
        losses = []
        for user, item, label in zip(users, items, batch_y):
            pred = self.net((user, item))
            losses.append(self.loss_func(pred, label).item())
            _, indices = pred.topk(k=100)
            predictions.append(item[indices].cpu().numpy())

        if log_loss_name:
            self.log(log_loss_name, np.mean(losses))

        return np.stack(predictions)

    def validation_step(self, batch: TBatch, batch_idx: int) -> np.ndarray:
        return self._validation_and_test_step(batch, "loss/val")

    def test_step(self, batch: TBatch, batch_idx: int) -> np.ndarray:
        return self._validation_and_test_step(batch)

    def validation_epoch_end(self, outputs: List[np.ndarray]) -> None:
        predictions = np.concatenate(outputs)
        gt = self.args.valid_dataset.gt[: predictions.shape[0]]
        mlb = self.args.valid_dataset.mlb

        n20 = get_ndcg(predictions, gt, mlb, top=20)
        r20 = get_recall(predictions, gt, mlb, top=20)

        results = {"val/n20": n20, "val/r20": r20}

        self.log_dict(results, prog_bar=True)
        self.should_prune(results["val/" + self.args.early_criterion])

    def test_epoch_end(self, outputs: List[np.ndarray]) -> None:
        predictions = np.concatenate(outputs)
        dataset = (
            self.args.valid_dataset if self.is_hptuning else self.args.test_dataset
        )
        gt = dataset.gt[: predictions.shape[0]]
        mlb = dataset.mlb
        n10 = get_ndcg(predictions, gt, mlb, top=10)
        n20 = get_ndcg(predictions, gt, mlb, top=20)
        r10 = get_recall(predictions, gt, mlb, top=10)
        r20 = get_recall(predictions, gt, mlb, top=20)

        results = {"test/n10": n10, "test/n20": n20, "test/r10": r10, "test/r20": r20}

        self.log_dict(results, prog_bar=True)


def init_run(*args, **kwargs):
    base_trainer.init_run(*args, **kwargs)


def check_args(args: AttrDict) -> None:
    early_criterion = ["loss", "n20", "r20"]
    model_name = ["MLP", "GMF", "NeuMF-end", "NeuMF-pre"]
    dataset_name = ["VG"]

    base_trainer.check_args(args, early_criterion, model_name, dataset_name)

    if args.model_name == "NeuMF-pre":
        assert (
            args.gmf_run_id is not None
        ), f"GMF_run_id must be set, when model_name = {args.model_name}"
        assert (
            args.mlp_run_id is not None
        ), f"MLP_run_id must be set, when model_name = {args.model_name}"

    if args.item_init_emb_path:
        assert args.model_name in [
            "NeuMF-end",
            "MLP",
        ], "If item_init_emb_path is set, model_name must be NeuMF-end or MLP"


def init_dataloader(args: AttrDict) -> None:
    base_trainer.init_dataloader(args, _get_dataset, _get_dataloader)

    logger.info(f"# of train dataset: {len(args.train_dataset):,}")
    logger.info(f"# of users: {args.num_user:,}")
    logger.info(f"# of items: {args.num_item:,}")


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
        NCFTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        NCFTrainerModel,
        metrics=["n10", "n20", "r10", "r20"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def inference(args: AttrDict):
    raise NotImplemented
