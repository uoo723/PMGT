"""
Created on 2022/01/05
@author Sangwoo Han
"""
import click

from pmgt.utils import log_elapsed_time


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option("--a", type=click.INT, default=3)
@log_elapsed_time
def train_ncf(**args):
    """Train for NCF"""


@cli.command(context_settings={"show_default": True})
@click.option("--b", type=click.FLOAT, default=0.4)
@log_elapsed_time
def train_dcn(**args):
    """Train for DCN"""


if __name__ == "__main__":
    cli()
