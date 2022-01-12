"""
Created on 2022/01/05
@author Sangwoo Han
"""
from typing import Dict, Union

import click
import numpy as np
from attrdict import AttrDict
from logzero import logger

from main import cli
from pmgt.utils import log_elapsed_time, save_args

# fmt: off

_common_options = [
    ########################################### Train Options #############################################
    click.option("--seed", type=click.INT, default=0, help="Seed for reproducibility"),
    click.option("--run-id", type=click.STRING, help="MLFlow Run ID for resume training"),
    click.option("--model-name", type=click.STRING, required=True, help="model name"),
    click.option("--dataset-name", type=click.STRING, required=True, help="dataset name"),
    click.option("--valid-size", default=0.2, help="validation dataset size"),
    click.option("--num-epochs", type=click.INT, default=20, help="training epochs"),
    click.option("--lr", type=click.FLOAT, default=1e-3, help="learning rate"),
    click.option("--decay", type=click.FLOAT, default=1e-2, help="Weight decay"),
    click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda"),
    click.option("--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"),
    click.option("--early", type=click.INT, default=5, help="Early stopping epoch"),
    click.option("--early-criterion", type=click.STRING, default="loss", help="Early stopping criterion"),
    click.option("--num-workers", type=click.INT, default=8, help="Number of workers for dataloader"),
    click.option("--train-batch-size", type=click.INT, default=256, help="train batch size"),
    click.option("--test-batch-size", type=click.INT, default=256, help="test batch size"),
    click.option("--gradient-max-norm", type=click.FLOAT, help="max norm for gradient clipping"),
    click.option("--accumulation-step", type=click.INT, default=1, help="accumlation step for small batch size"),
    click.option("--scheduler-warmup", type=click.FloatRange(0, 1), help="Ratio of warmup among total training steps"),
    click.option(
        "--scheduler-type",
        type=click.Choice(
            [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        ),
        help="Set type of scheduler",
    ),
    click.option(
        "--mode",
        type=click.Choice(["train", "eval", "inference"]),
        default="train",
        help="train: train and eval are executed. eval: eval only, inference: inference only",
    ),
    click.option("--inference-result-path",type=click.Path(), help="inference results path"),
    ######################################################################################################

    ########################################### Log Options ##############################################
    click.option("--experiment-name", type=click.STRING, default="baseline", help="experiment name"),
    click.option("--run-name", type=click.STRING, help="Set Run Name for MLFLow"),
    click.option("--tags", type=(str, str), multiple=True, help="set mlflow run tags"),
    click.option("--data-dir", type=click.Path(), default="./data", help="data directory"),
    click.option("--log-dir", type=click.Path(), default="./logs", help="log directory"),
    click.option("--run-script", type=click.Path(exists=True), help="Run script file path to log"),
    ######################################################################################################
]

# fmt: on


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@cli.command(context_settings={"show_default": True})
@add_options(_common_options)
@click.option(
    "--emb-dropout",
    type=click.FLOAT,
    default=0.0,
    help="dropout rate for embedding layer",
)
@click.option("--dropout", type=click.FLOAT, default=0.0, help="dropout rate")
@click.option(
    "--alpha",
    type=click.FLOAT,
    default=0.5,
    help="trade-off bewteen pretrained GMF and MLP",
)
@click.option(
    "--factor-num",
    type=click.INT,
    default=32,
    help="predictive factors numbers in the model",
)
@click.option("--num-layers", type=click.INT, default=3, help="number of layers in MLP")
@click.option(
    "--num-ng", type=click.INT, default=1, help="# of negative items for training"
)
@click.option(
    "--max-sample-items",
    type=click.INT,
    default=1000,
    help="Maximum # of items/user for testing",
)
@click.option(
    "--GMF-run-id",
    type=click.STRING,
    help="run id for GMF to load weights",
)
@click.option(
    "--MLP-run-id",
    type=click.STRING,
    help="run id for MLP to load weights",
)
@click.option(
    "--item-init-emb-path",
    type=click.Path(exists=True),
    help="item init embedding path from PMGT",
)
@click.option(
    "--freeze-item-init-emb",
    is_flag=True,
    default=False,
    help="freeze item init embedding affected only if item-init-emb-path is set",
)
@click.pass_context
@log_elapsed_time
def train_ncf(ctx: click.core.Context, **args):
    """Train for NCF"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    train_model("ncf", **args)


@cli.command(context_settings={"show_default": True})
@click.option("--b", type=click.FLOAT, default=0.4)
@click.pass_context
@log_elapsed_time
def train_dcn(ctx: click.core.Context, **args):
    """Train for DCN"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    train_dcn("dcn", **args)


@cli.command(context_settings={"show_default": True})
@add_options(_common_options)
@click.option(
    "--max-ctx-neigh",
    type=click.INT,
    default=5,
    help="maximum num of contextual neighbors",
)
@click.option(
    "--hop-sampling-sizes",
    type=click.INT,
    multiple=True,
    default=[16, 8, 4],
    help="# of maximum sampling nodes in each hop",
)
@click.option(
    "--max-total-samples",
    type=click.INT,
    default=10,
    help="maximum of total num of postive and negative nodes for each target node",
)
@click.option(
    "--min-neg-samples", type=click.INT, default=5, help="minimum num of negative nodes"
)
@click.option(
    "--hidden-size",
    type=click.INT,
    default=128,
    help="BERT output hidden size",
)
@click.option(
    "--intermediate-size",
    type=click.INT,
    default=128,
    help="BERT intermediate hidden size",
)
@click.option(
    "--num-hidden-layers",
    type=click.INT,
    default=5,
    help="BERT num of hidden layers",
)
@click.option(
    "--num-attention-heads",
    type=click.INT,
    default=1,
    help="BERT num of attention heads",
)
@click.option(
    "--beta",
    type=click.FLOAT,
    default=0.5,
    help="PMGT diversity promoting attention weight",
)
@click.option(
    "--random-node-ratio",
    type=click.FLOAT,
    default=0.2 * 0.1,
    help="PMGT random node ratio",
)
@click.option(
    "--mask-node-ratio",
    type=click.FLOAT,
    default=0.2 * 0.8,
    help="PMGT mask node ratio",
)
@click.pass_context
@log_elapsed_time
def train_pmgt(ctx: click.core.Context, **args):
    """Train for PMGT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    train_model("pmgt", **args)


def train_model(
    train_name, is_hptuning=False, **args
) -> Union[Dict[str, float], np.ndarray]:
    assert train_name in ["ncf", "pmgt"]

    args = AttrDict(args)

    if train_name == "ncf":
        import pmgt.ncf.trainer as trainer
    elif train_name == "pmgt":
        import pmgt.pmgt.trainer as trainer

    trainer.check_args(args)
    trainer.init_run(args)
    trainer.init_dataloader(args)
    trainer.init_model(args)

    if args.mode == "inference":
        logger.info("Inference mode")
        return trainer.inference(args)

    pl_trainer = None
    if args.mode == "train":
        _, pl_trainer = trainer.train(args)

    if args.mode == "eval":
        logger.info("Eval mode")

    try:
        return trainer.test(args, pl_trainer, is_hptuning=is_hptuning)
    except Exception as e:
        if pl_trainer:
            pl_logger = pl_trainer.logger
            pl_logger.experiment.set_terminated(pl_logger.run_id, status="FAILED")
        raise e
