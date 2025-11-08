import torch
from torch import nn
import torch.nn.functional as F
from collections.abc import Mapping
from typing import Callable, Generic,TypeVar
T = TypeVar("T")



class LazyDict(Mapping[str, T], Generic[T]):

    def __init__(self, factory: dict[str, Callable[[], T]]):
        self._factory = factory
        self._dict: dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        if key not in self._dict:
            if key not in self._factory:
                raise KeyError(key)
            self._dict[key] = self._factory[key]()
        return self._dict[key]

    def __setitem__(self, key: str, value: Callable[[], T]):
        self._factory[key] = value

    def __iter__(self):
        return iter(self._factory)

    def __len__(self):
        return len(self._factory)
    
_ACTIVATION_REGISTRY = LazyDict({
    "gelu":
    lambda: nn.GELU(),
    "gelu_pytorch_tanh":
    lambda: nn.GELU(approximate="tanh"),
    "relu":
    lambda: nn.ReLU(),
    "silu":
    lambda: nn.SiLU(),
})


def get_act_fn(act_fn_name: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_REGISTRY[act_fn_name]

