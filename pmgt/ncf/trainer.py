"""
Created on 2022/01/08
@author Sangwoo Han
"""
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
from attrdict import AttrDict
from logzero import logger
from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import get_scheduler

from ..callbacks import MLFlowExceptionCallback
from ..metrics import get_ndcg, get_recall
from ..ncf.datasets import NCFDataset
from ..ncf.models import NCF
from ..optimizers import DenseSparseAdamW
from ..utils import set_seed

TInput = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor]]
TOutput = torch.Tensor
TBatch = Tuple[TInput, TInput]


def _get_optimizer(args: AttrDict) -> Optimizer:
    model: nn.Module = args.model

    no_decay = ["bias", "LayerNorm.weight"]

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.decay,
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
    ]

    return DenseSparseAdamW(param_groups)


def _get_scheduler(args: AttrDict, optimizer: Optimizer) -> Optional[_LRScheduler]:
    if args.scheduler_type is None:
        return

    step_size = args.train_batch_size * args.accumulation_step
    num_training_steps = (
        (args.train_ids.shape[0] + step_size - 1) // step_size * args.num_epochs
    )
    num_warmup_steps = (
        int(args.scheduler_warmup * num_training_steps)
        if args.scheduler_warmup is not None
        else None
    )

    return get_scheduler(
        args.scheduler_type,
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )


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


def _load_pretrained_model(log_dir: str, run_id: str) -> NCF:
    ckpt_path, params = _get_ckpt_path(
        log_dir, run_id, load_best=True, return_params=True
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = NCF(
        int(params.num_user),
        int(params.num_item),
        int(params.factor_num),
        int(params.num_layers),
        float(getattr(params, "emb_dropout", 0.0)),
        float(params.dropout),
        params.model_name,
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
        GMF_model, MLP_model = None

    model = NCF(
        args.num_user,
        args.num_item,
        args.factor_num,
        args.num_layers,
        args.emb_dropout,
        args.dropout,
        args.model_name,
        GMF_model=GMF_model,
        MLP_model=MLP_model,
    )

    return model


def _get_ckpt_path(
    log_dir: str, run_id: str, load_best: bool = False, return_params: bool = False
) -> Union[str, Tuple[str, AttrDict]]:
    client = MlflowClient(log_dir)
    run = client.get_run(run_id)
    ckpt_root_dir = os.path.join(log_dir, run.info.experiment_id, run_id, "checkpoints")
    ckpt_path = os.path.join(ckpt_root_dir, "last.ckpt")
    if load_best:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        key = [k for k in ckpt["callbacks"].keys() if k.startswith("ModelCheckpoint")][
            0
        ]
        ckpt_path = ckpt["callbacks"][key]["best_model_path"]

    if return_params:
        return ckpt_path, AttrDict(run.data.params)

    return ckpt_path


class TrainerModel(pl.LightningModule):
    IGNORE_HPARAMS: List[str] = [
        "model",
        "loss_func",
        "train_dataset",
        "valid_dataset",
        "test_dataset",
        "train_dataloader",
        "valid_dataloader",
        "test_dataloader",
        "device",
        "run_script",
        "log_dir",
        "experiment_name",
        "data_dir",
        "eval_ckpt_path",
    ]

    def __init__(self, loss_func: Callable, **args: Any) -> None:
        super().__init__()
        self.args = AttrDict(args)
        self.net = self.args.model
        self.loss_func = loss_func
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    def forward(self, x: TInput) -> TOutput:
        return self.net(x)

    def configure_optimizers(self):
        optimizer = _get_optimizer(self.args)
        scheduler = _get_scheduler(self.args, optimizer)

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self):
        self.logger.experiment.set_tag(self.logger.run_id, "run_id", self.logger.run_id)
        self.logger.experiment.set_tag(self.logger.run_id, "host", os.uname()[1])
        logger.info(f"run_id: {self.logger.run_id}")

        if "run_script" in self.args:
            self.logger.experiment.log_artifact(
                self.logger.run_id, self.args.run_script, "scripts"
            )

    def on_train_epoch_start(self) -> None:
        if hasattr(self.args.train_dataset, "ng_sample") and self.global_step > 0:
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
        # print("predictions.shape", predictions.shape)
        # print("gt.shape", gt.shape)
        # print("mlb:", len(mlb.classes_))
        n20 = get_ndcg(predictions, gt, mlb, top=20)
        r20 = get_recall(predictions, gt, mlb, top=20)

        results = {"val/n20": n20, "val/r20": r20}

        self.log_dict(results, prog_bar=True)

    def test_epoch_end(self, outputs: List[np.ndarray]) -> None:
        predictions = np.concatenate(outputs)
        gt = self.args.test_dataset.gt[: predictions.shape[0]]
        mlb = self.args.test_dataset.mlb
        n10 = get_ndcg(predictions, gt, mlb, top=10)
        n20 = get_ndcg(predictions, gt, mlb, top=20)
        r10 = get_recall(predictions, gt, mlb, top=10)
        r20 = get_recall(predictions, gt, mlb, top=20)

        results = {"test/n10": n10, "test/n20": n20, "test/r10": r10, "test/r20": r20}

        self.log_dict(results, prog_bar=True)


def init_run(args: AttrDict) -> None:
    if args.seed is not None:
        logger.info(f"seed: {args.seed}")
        set_seed(args.seed)

    args.device = torch.device("cpu" if args.no_cuda else "cuda")
    args.num_gpus = torch.cuda.device_count()


def check_args(args: AttrDict) -> None:
    assert type(args.valid_size) in [float, int], "valid size must be int or float"
    if args.model_name == "NeuMF-pre":
        assert (
            args.gmf_run_id is not None
        ), f"GMF_run_id must be set, when model_name = {args.model_name}"
        assert (
            args.mlp_run_id is not None
        ), f"MLP_run_id must be set, when model_name = {args.model_name}"


def init_dataloader(args: AttrDict) -> None:
    logger.info(f"Dataset: {args.dataset_name}")
    train_dataset, valid_dataset, test_dataset = _get_dataset(args)
    args.train_dataset = train_dataset
    args.valid_dataset = valid_dataset
    args.test_dataset = test_dataset

    args.train_dataset.ng_sample()

    logger.info(f"# of train dataset: {len(train_dataset):,}")
    logger.info(f"# of users: {args.num_user:,}")
    logger.info(f"# of items: {args.num_item:,}")

    logger.info(f"Prepare Dataloader")

    train_dataloader, valid_dataloader, test_dataloader = _get_dataloader(
        args,
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    args.train_dataloader = train_dataloader
    args.valid_dataloader = valid_dataloader
    args.test_dataloader = test_dataloader


def init_model(args: AttrDict) -> None:
    logger.info(f"Model: NCF ({args.model_name})")

    model = _get_model(args)

    if args.num_gpus > 1 and not args.no_cuda:
        logger.info(f"Multi-GPU mode: {args.num_gpus} GPUs")
    elif not args.no_cuda:
        logger.info("single-GPU mode")
    else:
        logger.info("CPU mode")

    if args.num_gpus >= 1 and not args.no_cuda:
        gpu_info = [
            f"{i}: {torch.cuda.get_device_name(i)}" for i in range(args.num_gpus)
        ]
        logger.info("GPU info - " + " ".join(gpu_info))

    args.model = model


def train(args: AttrDict) -> Tuple[float, pl.Trainer]:
    if args.run_name is None:
        run_name = f"{args.model_cnf['name']}_{args.data_cnf['name']}"
    else:
        run_name = args.run_name

    mlf_logger = pl_loggers.MLFlowLogger(
        experiment_name=args.experiment_name, run_name=run_name, save_dir=args.log_dir
    )

    monitor = (
        "loss/val" if args.early_criterion == "loss" else f"val/{args.early_criterion}"
    )
    mode = "min" if args.early_criterion == "loss" else "max"

    early_stopping_callback = EarlyStopping(
        monitor=monitor, patience=args.early, mode=mode
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename=f"epoch={{epoch:02d}}-{monitor.split('/')[-1]}={{{monitor}:.4f}}",
        mode=mode,
        save_top_k=3,
        auto_insert_metric_name=False,
        save_last=True,
    )
    mlflow_exception_callback = MLFlowExceptionCallback()

    trainer_model = TrainerModel(loss_func=nn.BCEWithLogitsLoss(), **args)
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        gpus=args.num_gpus,
        precision=16 if args.mp_enabled else 32,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_max_norm,
        accumulate_grad_batches=args.accumulation_step,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
            mlflow_exception_callback,
        ],
        logger=mlf_logger,
    )

    ckpt_path = (
        _get_ckpt_path(args.log_dir, args.run_id, args.load_best)
        if args.run_id
        else None
    )
    trainer.fit(
        trainer_model, args.train_dataloader, args.valid_dataloader, ckpt_path=ckpt_path
    )

    args.eval_ckpt_path = checkpoint_callback.best_model_path

    best_score = checkpoint_callback.best_model_score
    best_score = best_score.item() if best_score else 0

    return best_score, trainer


def test(args: AttrDict, trainer: Optional[pl.Trainer] = None) -> None:
    if args.mode == "eval":
        assert args.run_id is not None, "run_id must be provided"
        ckpt_path = _get_ckpt_path(args.log_dir, args.run_id, True)
    else:
        ckpt_path = args.eval_ckpt_path

    trainer_model = TrainerModel(loss_func=nn.BCEWithLogitsLoss(), **args)

    trainer = trainer or pl.Trainer(
        gpus=args.num_gpus,
        precision=16 if args.mp_enabled else 32,
        enable_model_summary=False,
        logger=False,
    )

    results = trainer.test(
        trainer_model, args.test_dataloader, ckpt_path=ckpt_path, verbose=False
    )[0]
    metrics = ["n10", "n20", "r10", "r20"]

    msg = "\n" + "\n".join([f"{m}: {results['test/' + m]:.4f}" for m in metrics])
    logger.info(msg)
