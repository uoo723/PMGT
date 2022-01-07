"""
Created on 2022/01/08
@author Sangwoo Han
"""
from collections import deque
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from logzero import logger

TModel = Union[nn.DataParallel, nn.Module]


def clip_gradient(
    model: TModel,
    gradient_norm_queue: deque,
    gradient_clip_value: Optional[float] = None,
    verbose: bool = False,
):
    if gradient_clip_value is None:
        return

    max_norm = max(gradient_norm_queue)
    total_norm = nn.utils.clip_grad_norm_(
        model.parameters(), max_norm * gradient_clip_value
    )
    gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
    if total_norm > max_norm * gradient_clip_value:
        total_norm = total_norm.item() if hasattr(total_norm, "item") else total_norm
        max_norm = max_norm.item() if hasattr(max_norm, "item") else max_norm
        if verbose:
            logger.warning(
                f"Clipping gradients with total norm {round(total_norm, 5)} "
                f"and max norm {round(max_norm, 5)}"
            )


def swa_init(
    model: TModel, swa_state: Dict[str, torch.Tensor], verbose: bool = True
) -> None:
    if verbose:
        logger.info("SWA Initializing")

    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] = 1
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()


def swa_step(
    model: TModel, swa_state: Dict[str, torch.Tensor], verbose: bool = True
) -> None:
    if verbose:
        logger.info("SWA Step")

    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] += 1
    beta = 1.0 / swa_state["models_num"]
    with torch.no_grad():
        for n, p in model.named_parameters():
            swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)


def swap_swa_params(
    model: TModel, swa_state: Dict[str, torch.Tensor], verbose: bool = True
) -> None:
    if verbose:
        logger.info("SWA Swap Params")

    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    for n, p in model.named_parameters():
        p.data, swa_state[n] = swa_state[n], p.data
