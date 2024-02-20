import torch.optim as optim 

def adamw_optim_fn(network):
    return optim.AdamW(network.parameters())

default_optim_fn = adamw_optim_fn