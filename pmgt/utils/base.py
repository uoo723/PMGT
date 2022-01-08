"""
Created on 2022/01/05
@author Sangwoo Han
"""
import copy
import json
import os
import random
import time
from datetime import timedelta
from functools import wraps
from typing import Any, Dict, Union

import numpy as np
import torch
from attrdict import AttrDict
from logzero import logger


def log_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.info(f"elapsed time: {end - start:.2f}s, {timedelta(seconds=elapsed)}")

        return ret

    return wrapper


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_args(args: Union[Dict[str, Any], AttrDict], path: str) -> None:
    args = copy.deepcopy(args)
    args.pop("run_script")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(args, f, indent=4, ensure_ascii=False)
