"""
Created on 2022/01/05
@author Sangwoo Han
"""
import click
from attrdict import AttrDict

from pmgt.utils import log_elapsed_time


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option("--lr", type=click.FLOAT, default=1e-3, help="learning rate")
@click.option(
    "--decay",
    type=click.FLOAT,
    default=1e-2,
    help="Weight decay",
)
@click.option("--dropout", type=click.FLOAT, default=0.0, help="dropout rate")
@click.option(
    "--train-batch-size", type=click.INT, default=256, help="train batch size"
)
@click.option("--test-batch-size", type=click.INT, default=256, help="test batch size")
@click.option(
    "--num-workers", type=click.INT, default=8, help="Number of workers for dataloader"
)
@click.option("--num-epochs", type=click.INT, default=20, help="training epochs")
@click.option(
    "--gradient-max-norm",
    type=click.FLOAT,
    help="max norm for gradient clipping",
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
    "--early",
    type=click.INT,
    default=5,
    help="Early stopping epoch",
)
@click.option(
    "--early-criterion",
    type=click.Choice(["loss", "n20", "r20"]),
    default="n20",
    help="Early stopping criterion",
)
@click.option(
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
)
@click.option(
    "--scheduler-warmup",
    type=click.FloatRange(0, 1),
    help="Ratio of warmup among total training steps",
)
@click.option(
    "--accumulation-step",
    type=click.INT,
    default=1,
    help="accumlation step for small batch size",
)
@click.option(
    "--mode",
    type=click.Choice(["train", "eval"]),
    default="train",
    help="train: train and eval are executed. eval: eval only",
)
@click.option(
    "--model-name",
    type=click.Choice(["MLP", "GMF", "NeuMF-end", "NeuMF-pre"]),
    default="MLP",
    help="NCF model name",
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
@click.option("--dataset-name", type=click.Choice(["VG"]), default="VG", help="dataset")
@click.option("--valid-size", default=0.2, help="validation dataset size")
@click.option("--data-dir", type=click.Path(), default="./data", help="data directory")
@click.option("--log-dir", type=click.Path(), default="./logs", help="log directory")
@click.option(
    "--experiment-name", type=click.STRING, default="NCF", help="experiment name"
)
@click.option(
    "--run-script", type=click.Path(exists=True), help="Run script file path to log"
)
@click.option("--seed", type=click.INT, default=0, help="Seed for reproducibility")
@click.option("--run-name", type=click.STRING, help="Set Run Name for MLFLow")
@click.option("--run-id", type=click.STRING, help="MLFlow Run ID for resume training")
@click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda")
@click.option(
    "--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"
)
@log_elapsed_time
def train_ncf(**args):
    """Train for NCF"""
    import pmgt.ncf.trainer as trainer

    args = AttrDict(args)

    trainer.init_run(args)
    trainer.check_args(args)
    trainer.init_dataloader(args)
    trainer.init_model(args)

    pl_trainer = None
    if args.mode == "train":
        _, pl_trainer = trainer.train(args)

    try:
        trainer.test(args, pl_trainer)
    except Exception as e:
        if pl_trainer:
            logger = pl_trainer.logger
            logger.experiment.set_terminated(logger.run_id, status="FAILED")
        raise e


@cli.command(context_settings={"show_default": True})
@click.option("--b", type=click.FLOAT, default=0.4)
@log_elapsed_time
def train_dcn(**args):
    """Train for DCN"""


if __name__ == "__main__":
    cli()
