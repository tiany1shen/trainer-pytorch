import torch
import torch.nn as th_nn
import torch.optim as th_optim
import torch.utils.data as th_data

from pathlib import Path
from typing import Optional, Sequence, Tuple, cast, Callable
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from .module import SizedDataset, Network, LossManager
from .reproduce import RandomNumberState
from .plugin import PluginPriorityQueue, BasePlugin, ReinitNetworkWeightsPlugin
from .optim import default_optim_fn


class Trainer:
    __doc__ = r"""
    :class:`Trainer` 管理训练过程，对其的抽象如下：
    
    · 与训练相关的部分：设置为 Trainer 对象的属性实参
        - 训练轮次 epoch 
        - 半精度、混合精度设置 mixed-precision
        - 批数据规模 batch-size
        - 梯度累计次数 grad-acc
        - 随机种子 random-seed (用来生成 dataloader)
        - 计算设备 device
    
    · 与训练无关的部分：设置为 Trainer 对象的方法形参
        - 数据集 dataset
        - 网络模型 network
        - 训练损失 loss
        - 优化器 optimizer
    """
    def __init__(
        self, 
        exp_name: str, 
        epoch: int, 
        batch_size: int, 
        *,
        gradient_accumulation_step: int = 1, 
        init_random_seed: Optional[int] = None,
        device: str | int = "cpu",
        enable_auto_mixed_precision: bool = True,
        
        log_tool: Optional[str] = "tensorboard",
        log_dir: Optional[str] = None
    ) -> None:
        self.exp_name = exp_name
        # hyper params
        self.epoch_duration: int = epoch
        self.batch_size: int = batch_size
        self.gradient_accumulation_step: int = gradient_accumulation_step
        
        # local index counter, start from 1
        self._local_epoch: int 
        self._local_step: int
        self._local_iter: int
        
        self._start_epoch: int = 0
        
        # training state control
        self.rng = RandomNumberState(init_random_seed)
        self.device = torch.device(device)
        self.amp_enabled = enable_auto_mixed_precision
        
        self.check_setup()
        self.plugins = self.register_plugins([ReinitNetworkWeightsPlugin()])
        self.logger = self.register_logger(log_tool, log_dir)
    
    def check_setup(self) -> None:
        if self.batch_size % self.gradient_accumulation_step != 0:
            raise ValueError(f"Batch Size {self.batch_size} is not dividable by "
                            f"gradient accumulation step {self.gradient_accumulation_step}.")
        ...
    
    @property
    def step_start(self) -> bool:
        return not((self._local_iter - 1) % self.gradient_accumulation_step)
    
    @property
    def step_end(self) -> bool:
        return not(self._local_iter % self.gradient_accumulation_step)
    
    def train(
        self, *,
        dataset: th_data.Dataset, 
        network: th_nn.Module,
        losses: Tuple[Sequence, ...],    # zip(loss_names, loss_fns, loss_weights)
        optim_fn = default_optim_fn, # a function to build optimizer
    ) -> None:
        #! on-going
        #todo: TRAIN-LOOP 
        dataset = SizedDataset(dataset)
        network = Network(network, self.device)
        loss_fn = LossManager(*losses)
        grad_scaler = GradScaler(enabled=self.amp_enabled)
        
        optimizer: th_optim.Optimizer = optim_fn(network.unwrap_model)
        dataloader: th_data.DataLoader = self.build_dataloader(dataset)
        self._local_epoch: int = 0
        
        training_modules = {
            "trainer": self,
            "network": network,
            "optimizer": optimizer,
            "scaler": grad_scaler,
            "loss_fn": loss_fn
        }
        
        self.plugins.loop_beg_func(**training_modules)
        
        for _ in range(self.epoch_duration):
            
            self._local_epoch += 1
            self._local_iter: int = 0
            self._local_step: int = 0
            self.rng.update()
            
            if network.device != self.device:
                network.to(self.device)
            network.train()
            
            self.plugins.epoch_beg_func(**training_modules)
            
            for batch in dataloader:
                self._local_iter += 1
                
                if self.step_start:
                    self._local_step += 1
                    optimizer.zero_grad()
                    
                    self.plugins.step_beg_func(**training_modules)
                
                with autocast(enabled=self.amp_enabled):
                    loss_dict = loss_fn.compute_loss(network, batch)
                    loss_fn.store_values(loss_dict)
                    total_loss = loss_fn.summary_loss(loss_dict) / self.gradient_accumulation_step
                scaled_loss = cast(torch.Tensor, grad_scaler.scale(total_loss))
                scaled_loss.backward()
                
                if self.step_end:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    self.plugins.step_end_func(**training_modules)
                    
            self.plugins.epoch_end_func(**training_modules)
        self.plugins.loop_end_func(**training_modules)
    
    def build_dataloader(self, dataset: SizedDataset, num_workers: int = 4) -> th_data.DataLoader:
        # build a :class:`torch.utils.DataLoader` based on given dataset.
        mini_batch_size = self.batch_size // self.gradient_accumulation_step
        total_samples = len(dataset) // self.batch_size * self.batch_size
        self.dataloader_len = total_samples // mini_batch_size
        random_sampler = th_data.RandomSampler(dataset, num_samples=total_samples)
        return th_data.DataLoader(dataset, mini_batch_size, 
                            sampler=random_sampler, num_workers=num_workers)
    
    @property
    def epoch(self) -> int:
        return self._start_epoch + self._local_epoch
    
    @property
    def step(self) -> int:
        return  self._local_step + (self.epoch - 1) * \
            (self.dataloader_len // self.gradient_accumulation_step)
    
    @property
    def iter(self) -> int:
        return (self.epoch - 1) * self.dataloader_len + self._local_iter
    
    @property
    def random_seed(self) -> int:
        return self.rng.seed
    
    def state_dict(self):
        return {"epoch": self.epoch,
                "random_seed": self.random_seed}
    
    def load_state_dict(self, state_dict):
        self._start_epoch = state_dict['epoch']
        self.rng = RandomNumberState(state_dict['random_seed'])
    
    def register_logger(self, log_tool, log_dir):
        if log_tool is None:
            return None
        elif log_tool == "tensorboard":
            if log_dir is None:
                log_dir = Path("tb_log", self.exp_name)
            else:
                log_dir = Path(log_dir, self.exp_name)
            return SummaryWriter(log_dir)
    
    def register_plugins(self, plugins: Sequence[BasePlugin]) -> PluginPriorityQueue:
        for plug in plugins:
            plug.trainer = self 
        return PluginPriorityQueue(plugins)
    
    def append_plugin(self, plugin: BasePlugin):
        plugin.trainer = self
        self.plugins.append(plugin)
        return self
    
    def extend_plugins(self, plugins: Sequence[BasePlugin]):
        for plug in plugins:
            self.append_plugin(plug)
        return self
    
    @property
    def reinit_plugin(self) -> ReinitNetworkWeightsPlugin:
        return cast(ReinitNetworkWeightsPlugin, self.plugins[0])
    
    def enable_reinit(self, weight_file: Optional[str] = None):
        self.reinit_plugin.enable()
        self.reinit_plugin.register_weight_file(weight_file)
        return self
    
    def disable_reinit(self):
        self.reinit_plugin.disable()
        return self