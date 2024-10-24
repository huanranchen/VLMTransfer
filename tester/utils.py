import torch
from math import ceil
from torch import nn
from typing import List


def model_distribute(models: List[nn.Module]):
    """
    make each model on one gpu
    :return:
    """
    num_gpus = torch.cuda.device_count()
    models_each_gpu = ceil(len(models) / num_gpus)
    for i, model in enumerate(models):
        model.to(torch.device(f"cuda:{num_gpus - 1 - i // models_each_gpu}"))
        model.device = torch.device(f"cuda:{num_gpus - 1 - i // models_each_gpu}")
