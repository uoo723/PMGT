"""
Created on 2022/01/08
@author Sangwoo Han
"""
import copy
import json
import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import click
import optuna
from attrdict import AttrDict
from logzero import logger
from optuna import Study, Trial
from optuna.trial import TrialState
from ruamel.yaml import YAML

from main import cli
from pmgt.utils import log_elapsed_time
from train import _train_dcn, _train_ncf


def _load_train_params(config_filepath: str) -> AttrDict:
    with open(config_filepath, "r", encoding="utf-8") as f:
        return AttrDict(json.load(f))


def _get_hp_params(trial: Trial, hp_params: Dict):
    p = {}
    for key, value in hp_params.items():
        if value["type"] == "categorical":
            p[key] = trial.suggest_categorical(key, value["value"])
        elif value["type"] == "float":
            p[key] = trial.suggest_float(
                key, *value["value"], step=value.get("step", None)
            )
        elif value["type"] == "int":
            p[key] = trial.suggest_int(key, *value["value"])
    return p


def _max_trial_callback(study: Study, trial: Trial, n_trials: int) -> None:
    n_complete = len(
        [
            t
            for t in study.trials
            if t.state == TrialState.COMPLETE or t.state == TrialState.RUNNING
        ]
    )
    if n_complete >= n_trials:
        study.stop()


def objective(
    trial: Trial,
    train_params: Dict,
    hp_params: Dict,
    train_func: Callable,
    criterion: str,
) -> float:
    params = copy.deepcopy(train_params)
    params.update(_get_hp_params(trial, hp_params))
    params.tags = list(params.tags) + [("trial", trial.number)]
    results = train_func(is_hptuning=True, **params)
    return results[criterion]


@cli.command(context_settings={"show_default": True})
@click.option(
    "--hp-config-path",
    type=click.Path(exists=True),
    required=True,
    help="hp params config file path",
)
@click.option(
    "--train-config-path",
    type=click.Path(exists=True),
    required=True,
    help="train config file path",
)
@click.option("--n-trials", type=click.INT, default=20, help="# of trials")
@click.option("--study-name", type=click.STRING, default="study", help="Set study name")
@click.option(
    "--storage-path",
    type=click.Path(),
    default="./outputs/hpo_storage.db",
    help="Set storage path to save study",
)
@click.option(
    "--train-func",
    type=click.Choice(["train_ncf", "train_dcn"]),
    default="train_ncf",
    help="Set train function",
)
@log_elapsed_time
def hp_tuning(**args):
    """Hyper-parameter tuning"""
    args = AttrDict(args)

    yaml = YAML(typ="safe")
    hp_params = AttrDict(yaml.load(Path(args.hp_config_path)))
    train_params = _load_train_params(args.train_config_path)
    train_func = _train_ncf if args.train_func == "train_ncf" else _train_dcn
    storage_path = os.path.abspath(args.storage_path)

    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    train_params.tags = [("study_name", args.study_name)]

    direction = "minimize" if train_params.early_criterion == "loss" else "maximize"
    study: Study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction=direction,
    )

    try:
        study.optimize(
            partial(
                objective,
                train_params=train_params,
                hp_params=hp_params,
                train_func=train_func,
                criterion="test/" + train_params.early_criterion,
            ),
            callbacks=[partial(_max_trial_callback, n_trials=args.n_trials)]
        )
    except KeyboardInterrupt:
        logger.info("Stop tuning.")

    all_trials = sorted(
        study.trials, key=lambda x: x.value if x.value else 0, reverse=True
    )
    best_trial = all_trials[0]

    best_exp_num = best_trial.number
    best_score = best_trial.value
    best_params = best_trial.params

    logger.info(f"best_exp_num: {best_exp_num}")
    logger.info(f"best_score: {best_score}")
    logger.info(f"best_params:\n{best_params}")
