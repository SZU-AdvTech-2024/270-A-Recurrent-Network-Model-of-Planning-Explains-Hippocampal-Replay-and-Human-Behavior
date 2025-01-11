import gymnasium as gym
import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class LossHP():
    def __init__(self, βv, βe, βp, βr):
        self.βv = βv
        self.βe = βe
        self.βp = βp
        self.βr = βr