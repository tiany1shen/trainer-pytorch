import torch
import numpy 
import random

from typing import Optional


class RandomNumberState:
    r""" Class to manage epoch-level random seed. 
    
    """
    low: int = 0
    high: int = 2 ** 32 - 1
    
    def __init__(self, init_seed: Optional[int] = None) -> None:
        if init_seed is None:
            init_seed = self.generate_seed()
        self.seed: int = init_seed
        self.manual_seed()
    
    def update(self) -> None:
        self.seed = self.next_seed
        self.manual_seed()
    
    def generate_seed(self) -> int:
        return int(torch.randint(self.low, self.high, (1,)).item())
    
    def manual_seed(self, seed: Optional[int] = None) -> None:
        seed = self.seed if seed is None else seed
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        
        self.next_seed: int = self.generate_seed()

