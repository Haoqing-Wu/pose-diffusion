from functools import partial

import numpy as np
import torch

from posediff.utils.torch import build_dataloader


def build_dataloader_stack_mode(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,

):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
