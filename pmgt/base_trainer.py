"""
Created on 2022/01/11
@author Sangwoo Han
"""
import os
from typing import Any, Callable, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from attrdict import AttrDict
from logzero import logger
from mlflow.tracking import MlflowClient
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler
from mlflow.entities import Run
from .optimizers import DenseSparseAdamW
from .utils import set_seed


def get_optimizer(args: AttrDict) -> Optimizer:
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


def get_scheduler(args: AttrDict, optimizer: Optimizer) -> Optional[_LRScheduler]:
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


def get_run(log_dir: str, run_id: str) -> Run:
    client = MlflowClient(log_dir)
    run = client.get_run(run_id)
    return run


def get_ckpt_path(log_dir: str, run_id: str, load_best: bool = False) -> str:
    run = get_run(log_dir, run_id)
    ckpt_root_dir = os.path.join(log_dir, run.info.experiment_id, run_id, "checkpoints")
    ckpt_path = os.path.join(ckpt_root_dir, "last.ckpt")
    if load_best:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        key = [k for k in ckpt["callbacks"].keys() if k.startswith("ModelCheckpoint")][
            0
        ]
        ckpt_path = ckpt["callbacks"][key]["best_model_path"]

    return ckpt_path


class BaseTrainerModel(pl.LightningModule):
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
        "tags",
        "is_hptuning",
    ]

    def __init__(
        self,
        loss_func: Optional[Callable] = None,
        is_hptuning: bool = False,
        **args: Any,
    ) -> None:
        super().__init__()
        self.args = AttrDict(args)
        self.net = self.args.model
        self.loss_func = loss_func
        self.is_hptuning = is_hptuning
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.args)
        scheduler = get_scheduler(self.args, optimizer)

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self):
        self.logger.experiment.set_tag(self.logger.run_id, "run_id", self.logger.run_id)
        self.logger.experiment.set_tag(self.logger.run_id, "host", os.uname()[1])

        for k, v in self.args.tags:
            self.logger.experiment.set_tag(self.logger.run_id, k, v)

        logger.info(f"run_id: {self.logger.run_id}")

        if "run_script" in self.args:
            self.logger.experiment.log_artifact(
                self.logger.run_id, self.args.run_script, "scripts"
            )


def init_run(args: AttrDict) -> None:
    if args.seed is not None:
        logger.info(f"seed: {args.seed}")
        set_seed(args.seed)

    args.device = torch.device("cpu" if args.no_cuda else "cuda")
    args.num_gpus = torch.cuda.device_count()
