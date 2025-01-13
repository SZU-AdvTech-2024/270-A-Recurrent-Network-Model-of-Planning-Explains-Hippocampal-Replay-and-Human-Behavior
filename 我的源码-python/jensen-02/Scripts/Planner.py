import numpy as np
import torch

class Planner:
    def __init__(self, plan_depth: int, maze_size: int, batch_size: int):
        self.plan_in_size = 0
        self.plan_out_size = 0
        self.maze_size = maze_size
        self.state_size = maze_size * maze_size
        self.plan_depth = plan_depth
        self.batch_size = batch_size
        
        if plan_depth > 0 :
            # plan plan_depth 个动作的序列 + 能否到达奖励位置
            self.plan_in_size = 4 * self.plan_depth + 1
            # 输出是reward的位置
            self.plan_out_size = self.state_size
    
        self.plan_cache = torch.zeros((batch_size, 1, self.plan_in_size))

    def store_plan_cache(self, plan_path, found_rew):
        # flatten plan_path的第二、三维度，并在最后cat上found_rew
        # 展平 plan_path 的第二、三维度
        flattened_plan_path = plan_path.view(self.batch_size, -1)
        
        # 拼接 found_rew
        plan_cache = torch.cat((flattened_plan_path, found_rew.unsqueeze(1)), dim=-1)
        
        # 更新 plan_cache
        self.plan_cache = plan_cache.unsqueeze(1)
        
    def get_plan_cache(self):
        return self.plan_cache
        
    def clear_plan_cache(self):
        self.plan_cache = torch.zeros((self.batch_size, 1, self.plan_in_size))