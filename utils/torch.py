import torch

from typing import List


def combine_module_params(*modules: List[torch.nn.Module]) -> torch.nn.Parameter:
    params = []
    for module in modules:
        params += module.parameters()
    return params


def join_params(*parameters: List[torch.nn.Parameter]) -> torch.nn.Parameter:
    params = []
    for param in parameters:
        params += param
    return params
