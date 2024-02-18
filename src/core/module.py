import torch as th
import torch.nn as th_nn
import torch.utils.data as th_data

from copy import deepcopy
from collections import deque
from functools import cached_property
from typing import TypeVar, Generic, cast, Sequence, Mapping, Callable, Optional, Union, Sized
from typing_extensions import TypeAlias

#===============================================================================
#   Dataset
#===============================================================================
""" 
A dataset is a repository of training/evaluating data. 

In our framework, we use the mapping-style :class:`torch.utils.data.Dataset`to manage data.

To access to a data sample by its index, use :meth:`__getitem__`. Data samples are 
all of type :class:`torch.Tensor` or variants, the most commonly used types are 
tuple of tensor or dictionary of tensor.

By default, :meth:`__len__` should also be overwrited for indicating the size of dataset, 
but it is fine to assign other number to it since its main functional is to set the length of 
training epoch correct.
"""

class SizedDataset(th_data.Dataset):
    """An abstract class representing a :class:`Dataset`.
    
    All datasets must be assigned with a finite `size` by overwrite :meth:`__len__`, 
    which helps setting a correct :class:`torch.utils.data.DataLoader`. 
    """
    def __init__(self, dataset: th_data.Dataset):
        super().__init__()
        self.dataset = dataset
    
    def __getitem__(self, index: int):
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

#===============================================================================
#   Network
#===============================================================================
"""
A Network is a parametric model that process tensor features.

In our framework, we use the :class:`torch.nn.Module` to manage parameters, buffers
and the forward computation.

In different to normal :class:`torch.nn.Module`, do not define the model in :meth:`__init__`, 
but in another method :meth:`init`, which will be called in the former initialization.

A class method :meth:`init_parameters_weight` is called for initializing parameters and buffers. 
Most built-in modules in `PyTorch` already have a method called :meth:`reset_parameters`
doing the same things.
"""

class Network(th_nn.Module):
    """An abstract class representing a neural network.
    
    Overwrite :meth:`init` method (instead of :meth:`__init__`) to define the network.
    
    Attributes:
        :meth:`init_parameters_weight` can be called explicitly to reset parameters. 
            If the :kwarg:`state_dict`(by default: `None`) argument is given, the model 
            will load the weights in it. 
        :attr:`device` return the a :class:`torch.device` object on which the model 
            is located. If the model has no tensor parameters or buffers, a `NameError` 
            will be raised.
    """
    
    def __init__(self, network: th_nn.Module, device: Optional[Union[th.device, str, int]] = None) -> None:
        self.network = deepcopy(network)
        
        if device is not None:
            self.network = self.network.to(device)
    
    @classmethod
    def recurrent_reset_parameters(cls, module: th_nn.Module) -> None:
        for sub_module in module.children():
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters() # type: ignore
            else:
                cls.recurrent_reset_parameters(sub_module)
    
    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)
    
    def init_weight(self, *, 
                    state_dict: Optional[dict] = None, strict: bool = True, 
                    device: Optional[th.device] = None) -> None:
        if state_dict is not None:
            self.network.load_state_dict(state_dict, strict)
        else:
            self.recurrent_reset_parameters(self.network)
        if device is not None:
            self.to(device)

    def state_dict(self):
        return self.network.state_dict()
    
    @property
    def device(self) -> th.device:
        return next(self.network.parameters()).device
    
    @property
    def unwrap_model(self) -> th_nn.Module:
        return self.network

#===============================================================================
#   Losses & Metrics
#===============================================================================
""" 
We concern about scalar values like loss values or metrics during training/evaluating 
process. Abstract class :class:`ScalarManager` defined below tells the trainer 
which scalar values should be recorded and how to compute them.

We offer two typical subclasses for loss value and evaluating metrics respectively:
    - :class:`LossManager`
    - :class:`MetricManager`
"""
T_val = TypeVar('T_val', float, th.Tensor)


class Tracker(Generic[T_val]):
    def __init__(self, name: str, maxlen: int = 100, *,
                smooth_fn: Callable[[Sequence[T_val]], float] = lambda x: float(x[-1]), 
                store_fn: Callable[[T_val], T_val] = lambda x:x) -> None:
        self.name: str = name 
        self.smooth_fn: Callable[[Sequence[T_val]], float] = smooth_fn
        self.store_fn: Callable[[T_val], T_val] = store_fn
        
        self.cache: deque[T_val] = deque(maxlen=maxlen)
    
    @property
    def storage(self) -> Sequence[T_val]: return list(self.cache)
        
    def store(self, value: T_val) -> None:
        self.cache.append(self.store_fn(value))
    
    def extract(self) -> float:
        if len(self.cache) == 0:
            raise RuntimeError(f"Try to extract element from :{self.__class__.__name__}:`{self.name}`, whose size is {len(self.cache)}!")
        return self.smooth_fn(self.cache)

class LossTracker(Tracker[th.Tensor]):
    @classmethod
    def avg_tensor(cls, tensors: Sequence[th.Tensor]) -> float:
        avg_val = th.tensor(tensors).mean()
        return avg_val.float().item()
    
    @classmethod
    def store_tensor(cls, tensor: th.Tensor) -> th.Tensor:
        return tensor.detach().mean()
    
    def __init__(self, name: str, maxlen: int = 100) -> None:
        super().__init__(name, maxlen=maxlen, smooth_fn=self.avg_tensor, store_fn=self.store_tensor)

class MetricTracker(Tracker[float]):
    def __init__(self, name: str, maxlen: int = 100) -> None:
        super().__init__(name, maxlen=maxlen)


class ScalarManager(Generic[T_val]):
    """ A abstract class to register and compute scalar values."""
    scalar_type: str = 'scalar'
    tracker_type = Tracker
    
    def __init__(self, scalar_names: Sequence[str], *,
                tracker_maxlen: int = 100) -> None:
        self.scalar_names: list[str] = list(scalar_names)
        self.is_compatible: bool = False
        
        self.trackers: dict[str, Tracker] = {}
        self.register_trackers(maxlen=tracker_maxlen)
    
    def check_result(self, scalar_dict: Mapping[str, T_val]) -> None:
        if self.is_compatible:
            return
        unexpected_keys = list(set(scalar_dict) - set(self.scalar_names))
        missing_keys = list(set(self.scalar_names) - set(scalar_dict))
        error_msgs: list[str] = []
        if len(unexpected_keys) > 0:
            error_msgs.append(f"Unexpected {self.scalar_type}(s): {', '.join(k for k in unexpected_keys)}")
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing {self.scalar_type}(s): {', '.join(k for k in missing_keys)}")
        if len(error_msgs) > 0:
            raise RuntimeError(f"Error(s) in checking {self.scalar_type} results: \n\t"
                            "{}\nRevise :meth:`compute` to return compatible results.".format('\n\t'.join(error_msgs)))
        self.is_compatible = True
        return
    
    def register_tracker(self, name: str, maxlen: int, *arg, **args) -> None: 
        if name in self.trackers:
            raise KeyError(f"Key {name} has already been registered.")
        self.trackers[name] = self.tracker_type(name, maxlen=maxlen)
    
    def register_trackers(self, maxlen: int) -> None:
        for name in self.scalar_names:
            self.register_tracker(name, maxlen)
    
    def store_values(self, value_dict: Mapping[str, T_val]) -> None:
        for name in self.trackers:
            self.trackers[name].store(value_dict[name])
    
    def extract_values(self) -> Mapping[str, float]:
        scalar_dict = {}
        for name in self.trackers:
            scalar_dict[name] = self.trackers[name].extract()
        return scalar_dict
    
    @property
    def storage(self) -> Mapping[str, Sequence[T_val]]:
        return {name: self.trackers[name].storage for name in self.scalar_names}


Batch: TypeAlias = Union[th.Tensor, Sequence[th.Tensor], Mapping[str, th.Tensor]]

def move_batch(batch: Batch, device: th.device):
    if isinstance(batch, th.Tensor):
        return batch.to(device)
    if isinstance(batch, Sequence):
        return [value.to(device) if isinstance(value, th.Tensor) else value for value in batch]
    if isinstance(batch, Mapping):
        return {key:value.to(device) if isinstance(value, th.Tensor) else value for key, value in batch.items()}

class LossManager(ScalarManager[th.Tensor]):
    scalar_type: str = "loss"
    tracker_type = LossTracker
    
    def __init__(self, loss_names: Sequence[str], 
                loss_fns: Sequence[Callable],
                loss_weights: Optional[Sequence[float]] = None,
                *,
                tracker_maxlen: int = 100) -> None:
        super().__init__(loss_names, tracker_maxlen=tracker_maxlen)
        self.loss_fns: list[Callable] = list(loss_fns)
        if loss_weights is None:
            loss_weights = [1.0 for k in loss_names]
        self.loss_weights: Sequence[float] = list(loss_weights)
    
    @property
    def loss_names(self) -> list[str]:
        return self.scalar_names
    
    def compute_loss(self, network: Network, batch_input: Batch) -> Mapping[str, th.Tensor]:
        batch_input = move_batch(batch_input, network.device)
        loss_dict = {}
        for name, loss_fn in zip(self.loss_names, self.loss_fns):
            loss_dict[name] = loss_fn(network, batch_input)
        return loss_dict
    
    def summary_loss(self, loss_dict: Mapping[str, th.Tensor]) -> th.Tensor:
        self.check_result(loss_dict)
        total_loss = th.tensor(0.)
        for k, w in zip(self.loss_names, self.loss_weights):
            total_loss += loss_dict[k].mean() * w 
        return total_loss


class MetricManager(ScalarManager[float]):
    scalar_type: str = "metric"
    tracker_type = MetricTracker
    
    def __init__(self, metric_names: Sequence[str], *, tracker_maxlen: int = 100) -> None:
        super().__init__(metric_names, tracker_maxlen=tracker_maxlen)
    
    @property
    def metric_names(self) -> list[str]:
        return self.scalar_names

