from pathlib import Path
from bisect import insort
from typing import TYPE_CHECKING, Optional, Literal, Sequence
from typing_extensions import TypedDict, NotRequired, override

import torch as th
from torch.optim import Optimizer 
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from .module import Network, LossManager

if TYPE_CHECKING:
    from .trainer import Trainer

class HookPriority(TypedDict):
    priority: int
    description: NotRequired[str]

class HookInfo(TypedDict):
    loop_beg: NotRequired[HookPriority]
    loop_end: NotRequired[HookPriority]
    epoch_beg:NotRequired[HookPriority]
    epoch_end:NotRequired[HookPriority]
    step_beg: NotRequired[HookPriority]
    step_end: NotRequired[HookPriority]
    
class BasePlugin:
    trainer: 'Trainer'
    hook_info: HookInfo
    
    def __init__(self, hook_info) -> None: self.hook_info = hook_info
    def loop_beg_func(self, *args, **kwargs): ... 
    def loop_end_func(self, *args, **kwargs): ...
    def epoch_beg_func(self, *args, **kwargs):...
    def epoch_end_func(self, *args, **kwargs):...
    def step_beg_func(self, *args, **kwargs): ...
    def step_end_func(self, *args, **kwargs): ...
    
    @property
    def name(self) -> str: return self.__class__.__name__[:-6]  # all subclass name end with 'Plugin'
    
    @property
    def state(self):
        return NotImplemented
    
    def __repr__(self) -> str:
        return f"{self.name} Plugin at {hex(id(self))}:\n" + \
            "\n".join(f"\t{key}: {value}" for key, value in self.state.items())
    
    @property
    def log_prefix(self) -> str: return f"[PLUGIN]({self.name})"
    
    def log(self, message: str) -> None:
        print(self.log_prefix + ' ' + message)


class PluginPriorityQueue:
    hook_stage: dict[str, str] = {
        "loop_beg": "Loop Begin", 
        "loop_end": "Loop End",
        "epoch_beg": "Epoch Begin",
        "epoch_end": "Epoch End",
        "step_beg": "Step Begin",
        "step_end": "Step End"
    }
    def __init__(self, plugins: Optional[Sequence[BasePlugin]] = None) -> None:
        if plugins is None:
            plugins = []
        self.plugins = list(plugins)
        self.sorted_priority_indice = self.sort_indices()
    
    def sort_indices(self) -> dict[str, list[int]]:
        indices = {hook: list() for hook in self.hook_stage}
        
        for i, plug in enumerate(self.plugins):
            for hook in plug.hook_info:
                indices[hook].append(i)
        
        for hook in indices:
            indices[hook].sort(key=lambda i: self.plugins[i].hook_info[hook]["priority"])
        
        return indices
    
    def loop_beg_func(self, *args, **kwargs) -> None:
        for i in self.sorted_priority_indice["loop_beg"]:
            self.plugins[i].loop_beg_func(*args, **kwargs)
    
    def loop_end_func(self, *args, **kwargs):
        for i in self.sorted_priority_indice["loop_end"]:
            self.plugins[i].loop_end_func(*args, **kwargs)
    
    def epoch_beg_func(self, *args, **kwargs) -> None:
        for i in self.sorted_priority_indice["epoch_beg"]:
            self.plugins[i].epoch_beg_func(*args, **kwargs)
    
    def epoch_end_func(self, *args, **kwargs) -> None:
        for i in self.sorted_priority_indice["epoch_end"]:
            self.plugins[i].epoch_end_func(*args, **kwargs)
    
    def step_beg_func(self, *args, **kwargs) -> None:
        for i in self.sorted_priority_indice["step_beg"]:
            self.plugins[i].step_beg_func(*args, **kwargs)
    
    def step_end_func(self, *args, **kwargs) -> None:
        for i in self.sorted_priority_indice["step_end"]:
            self.plugins[i].step_end_func(*args, **kwargs)
    
    def __getitem__(self, index: int):
        return self.plugins[index]
    
    def __repr__(self) -> str:
        string = "Hook Plugins including:"
        for hook, stage in self.hook_stage.items():
            indice = self.sorted_priority_indice[hook]
            if len(indice) > 0:
                string += "\n" + stage + " Hook(s):\n"
                for i in indice:
                    plug = self.plugins[i]
                    info = plug.hook_info[hook]
                    string += f"  [{info['priority']}]  " + plug.name
                    if 'description' in info:
                        string += ": " + info['description']
                    string += "\n"
        return string
    
    def append(self, plugin: BasePlugin) -> None:
        index = len(self.plugins)
        self.plugins.append(plugin)
        for hook in plugin.hook_info:
            insort(self.sorted_priority_indice[hook], index, 
                   key=lambda i: self.plugins[i].hook_info[hook]["priority"])
    
    def extend(self, plugins: list[BasePlugin]) -> None:
        for plugin in plugins:
            self.append(plugin)


class ReinitNetworkWeightsPlugin(BasePlugin):
    def __init__(self, weight_file: Optional[str] = None, strict: bool = True):
        hook_info: HookInfo = {
            "loop_beg": {
                "priority": 2,
                "description": "Re-initialize network weights randomly or from an assigned file."
            }
        }
        super().__init__(hook_info)
        self.enabled: bool = True
        self.weight_file: Optional[Path] = None
        self.register_weight_file(weight_file)
        self.strict = strict
    
    def register_weight_file(self, weight_file: Optional[str]):
        if weight_file is not None:
            self.weight_file = Path(weight_file)
    
    @override
    def loop_beg_func(self, 
                    network: Network, 
                    *args, **kwargs):
        if self.enabled:
            if self.weight_file is not None:
                self.log(f"Re-initialize network weights from '{self.weight_file}'")
            else:
                self.log("Re-initialize network weights randomly.")
            network.init_weight(state_dict=self.get_state_dict(), strict=self.strict, device=self.trainer.device)
        else:
            self.log("Do not re-initialize network weights again.")
    
    def get_state_dict(self):
        if self.weight_file is not None:
            return th.load(self.weight_file)
        else:
            return None
    
    def disable(self) -> None:
        self.enabled = False
        
    def enable(self) -> None:
        self.enabled = True
    
    @property
    def state(self):
        return {"enabled": self.enabled,
                "re-init from": self.weight_file}


class LoadTrainerStatePlugin(BasePlugin):
    def __init__(self, checkpoint_path: Optional[str] = None, 
                network_file: Optional[str] = None,
                optimizer_file: Optional[str] = None,
                scaler_file: Optional[str] = None,
                strict: bool = True) -> None:
        hook_info: HookInfo = {
            "loop_beg": {
                "priority": 1, 
                "description": "Load `state_dict` for modules."}
            
        }
        super().__init__(hook_info)
        self.trainer_file: Optional[Path] = None
        self.network_file: Optional[Path] = None
        self.optimizer_file: Optional[Path] = None
        self.scaler_file: Optional[Path] = None
        self.strict: bool = strict
        
        if checkpoint_path:
            self.trainer_file = Path(checkpoint_path, "trainer_state_dict.pth")
            self.network_file = Path(checkpoint_path, "network_state_dict.pth")
            self.optimizer_file = Path(checkpoint_path, "optimizer_state_dict.pth")
            self.scaler_file = Path(checkpoint_path, "scaler_state_dict.pth")
        else:
            if network_file is not None:
                self.network_file = Path(network_file)
            if optimizer_file is not None:
                self.optimizer_file = Path(optimizer_file)
            if scaler_file is not None:
                self.scaler_file = Path(scaler_file)
        
    __doc__ = r""" 
    分别加载 trainer, network, optimizer, grad scaler 四个 module 的状态
    
    1. 如果初始化时给出了 checkpoint_path 参数，则其他参数不起作用，state_dict 文件将从 
        checkpoint_path 目录下面自动寻找。
    2. 未给出 checkpoint_path 参数时，可以指定其他路径参数，对应的 module 将会加载相应路径的文件。
    3. 如果给出 network_file 参数，则关闭 trainer 的 init_network_weight 开关，在相关 hook 
        中不在随机初始化网络权重。
    """
    
    @override
    def loop_beg_func(self, 
                    trainer: 'Trainer', 
                    network: Network, 
                    optimizer: Optimizer,
                    scaler: GradScaler,
                    *args, **kwargs) -> None:
        if self.trainer_file is not None:
            self.log(f"Loading checkpoint from '{self.trainer_file.parent}'.")
            trainer.load_state_dict(th.load(self.trainer_file))
        
        if self.network_file is not None:
            self.log(f"Loading network weights from '{self.network_file}'.")
            network.init_weight(state_dict=th.load(self.network_file), strict=self.strict, device=self.trainer.device)
            self.trainer.disable_reinit()
        
        if self.optimizer_file is not None:
            self.log(f"Loading optimizer state from '{self.optimizer_file}'.")
            optimizer.load_state_dict(th.load(self.optimizer_file))
        
        if self.scaler_file is not None:
            self.log(f"Loading grad scaler state from '{self.scaler_file}'.")
            scaler.load_state_dict(th.load(self.scaler_file))

    @property
    def state(self):
        return {
            "trainer": self.trainer_file,
            "network": self.network_file,
            "optimizer": self.optimizer_file,
            "scaler": self.scaler_file
        }

class SavePlugin(BasePlugin):
    def __init__(self, save_dir: str, 
                period: int, hook_info: HookInfo) -> None:
        super().__init__(hook_info)
        self.save_dir: Path = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.period: int = period
    
    def save_trainer(self, trainer: 'Trainer', save_path: Path):
        self.log(f"Saving trainer to '{save_path}'")
        th.save(trainer.state_dict(), save_path)
        return self
    
    def save_network(self, network: Network, save_path: Path):
        self.log(f"Saving network to '{save_path}'")
        th.save(network.unwrap_model.state_dict(), save_path)
        return self
    
    def save_optimizer(self, optimizer: Optimizer, save_path: Path):
        self.log(f"Saving optimizer to '{save_path}'")
        th.save(optimizer.state_dict(), save_path)
        return self
    
    def save_scaler(self, scaler: GradScaler, save_path: Path):
        self.log(f"Saving scaler to '{save_path}'")
        th.save(scaler.state_dict(), save_path)
        return self
    
    def save(self, save_dir: Path,
            trainer: 'Trainer',
            network: Network,
            optimizer: Optimizer,
            scaler: GradScaler,
            *args, **kwargs) -> None:
        (
        self
        .save_trainer(trainer, save_dir / "trainer_state_dict.pth")
        .save_network(network, save_dir / "network_state_dict.pth")
        .save_optimizer(optimizer, save_dir / "optimizer_state_dict.pth")
        .save_scaler(scaler, save_dir / "scaler_state_dict.pth")
        )
    
    @property
    def state(self):
        return {
            "save diretory": self.save_dir,
            "save period": self.period
        }

class EpochSavePlugin(SavePlugin):
    def __init__(self, save_dir: str, period: int) -> None:
        hook_info: HookInfo = {
            "epoch_end": {
                "priority": 5,
                "description": f"Save training state every {period} epochs."}
        }
        super().__init__(save_dir, period, hook_info)
    
    def is_enable_epoch(self) -> bool:
        return self.trainer.epoch % self.period == 0
    
    @override
    def epoch_end_func(self, *args, **kwargs) -> None:
        if self.is_enable_epoch():
            epoch_save_dir: Path = self.save_dir / f"epoch-{self.trainer.epoch}"
            epoch_save_dir.mkdir(parents=True, exist_ok=True)
            self.save(epoch_save_dir, *args, **kwargs)
    
    @property
    @override
    def state(self):
        state = super().state
        state.update({"save level": "epoch"})
        return state

class StepSavePlugin(SavePlugin):
    def __init__(self, save_dir: str, period: int) -> None:
        hook_info: HookInfo = {
            "step_end": {
                "priority": 5,
                "description": f"Save training state every {period} steps."}
        }
        super().__init__(save_dir, period, hook_info)
    
    def is_enable_step(self) -> bool:
        return self.trainer.step % self.period == 0
    
    @override
    def step_end_func(self, *args, **kwargs) -> None:
        if self.is_enable_step():
            epoch_save_dir: Path = self.save_dir / f"step-{self.trainer.step}"
            epoch_save_dir.mkdir(parents=True, exist_ok=True)
            self.save(epoch_save_dir, *args, **kwargs)
    
    @property
    @override
    def state(self):
        state = super().state
        state.update({"save level": "step"})
        return state

class AdjustLearningRatePlugin(BasePlugin):
    """ 
    `lr_adjust_fn` args:
        lr: learning rate
        epoch_index: epoch index
        param_group_index: i
    
    Examples:
    
    `lr` decreases exponentially every epoch:
    >>> def exponential_desent_fn(lr, epoch_index, param_group_index) -> float:
    ...     return lr * 0.999
    
    `lr` becomes half every 5 epochs (do not forget to keep it unchange in other epoch):
    >>> def half_every_5_epoch_fn(lr, epoch_index, param_group_index) -> float:
    ...     if epoch_index % 5 == 1:
    ...         return lr * 0.5
    ...     else:
    ...         return lr
    
    only change the first param_group `lr`:
    >>> def adjust_1st_fn(lr, epoch_index, param_group_index) -> float:
    ...     if param_group_index == 0:
    ...         return lr * 0.999
    ...     else:
    ...         return lr
    """
    def __init__(self, lr_adjust_fn) -> None:
        hook_info: HookInfo = {
            "epoch_beg": {
                "priority": 5,
                "description": "Adjust learning rate."
            }
        }
        super().__init__(hook_info)
        self.adjust_fn = lr_adjust_fn
    
    @override
    def epoch_beg_func(self, optimizer: Optimizer, *args, **kwargs):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = self.adjust_fn(
                lr=param_group['lr'],
                epoch_index=self.trainer.epoch,
                param_group_index=i
            )
    
    @property
    @override
    def state(self):
        return {}


def repr_number(num):
    if num < 1_000:
        return f"{num:>3}"
    elif num < 1_000_000:
        return f"{num/1000:>4} K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:>4} M"
    else:
        return f"{num/1_000_000_000:>4} B"


class LossLoggerPlugin(BasePlugin):
    def __init__(self, period: int):
        hook_info: HookInfo = {
            "step_end": {
                "priority": 5,
                "description": f"Log message every {period} steps."
            },
            "loop_end": {
                "priority": 5,
                "description": "close writer."
            }
        }
        super().__init__(hook_info)
        self.period: int = period
    
    def is_enable_step(self) -> bool:
        return self.period == 1 or self.trainer.step % self.period == 1
    
    @override
    def step_end_func(self, loss_fn: LossManager, *args, **kwargs):
        if self.is_enable_step():
            loss_dict = loss_fn.extract_values()
            data_fed = self.trainer.batch_size * (self.trainer.step-1)
            if self.trainer.logger is None:
                msg = f"|{repr_number(data_fed)}| " 
                msg += " | ".join(f"{name}_loss: {value:.2e}" for name, value in loss_dict.items())
                self.log(msg)
            elif isinstance(self.trainer.logger, SummaryWriter):
                total_loss = 0
                for name, weight in zip(loss_dict, loss_fn.loss_weights):
                    tag = f"loss/{name}"
                    value = loss_dict[name]
                    total_loss += value * weight
                    self.trainer.logger.add_scalar(tag, value, data_fed)
                if len(loss_dict) > 1:
                    self.trainer.logger.add_scalar("loss/total", total_loss, data_fed)
    
    @override
    def loop_end_func(self, *args, **kwargs):
        if isinstance(self.trainer.logger, SummaryWriter):
            self.trainer.logger.close()
        
    