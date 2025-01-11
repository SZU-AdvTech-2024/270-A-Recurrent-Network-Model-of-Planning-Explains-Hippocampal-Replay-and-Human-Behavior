import numpy as np
from scipy.stats import rv_discrete
from Scripts.Maze import maze
import torch

@torch.no_grad()
def state_from_loc(Larena, loc):
    return np.array([loc // Larena, loc % Larena])

def reset_agent_state(Larena, reward_location, batch):
    Nstates = Larena**2
    # 创建均匀分布的类别分布
    probabilities = np.ones(Larena) / Larena
    categorical = rv_discrete(values=(np.arange(Larena), probabilities))
    
    # 随机生成起始位置 (batch, 2)
    agent_state = categorical.rvs(size=(batch, 2))
    
    if reward_location is not None:
        # 确保不能从奖励位置开始
        for b in range(batch):
            tele_reward_location = np.ones(Nstates) / (Nstates - 1)
            tele_reward_location[reward_location[b][0] * Larena + reward_location[b][1]] = 0
            categorical = rv_discrete(values=(np.arange(Nstates), tele_reward_location))
            ct = categorical.rvs(size=(1, 1))
            agent_state[b] = state_from_loc(Larena, ct[0][0])
    
    return agent_state.astype(np.int32)

@torch.no_grad()
def gen_maze_walls(Larena, batch):
    wall_loc = np.zeros((batch, Larena**2, 4), dtype=np.float32)
    for b in range(batch):
        wall_loc[b, :, :] = maze(Larena)
    return wall_loc

@torch.no_grad()
def initialize_arena(Larena, batch):
    # 生成迷宫墙
    wall_loc = gen_maze_walls(Larena, batch)
    
    # 生成奖励位置
    reward_location = reset_agent_state(Larena, None, batch)
    
    # 生成起始位置
    agent_state = reset_agent_state(Larena, reward_location, batch)
    
    return wall_loc, reward_location, agent_state


if __name__ == "__main__":
    # gen_maze_walls(4, 2)
    # reset_agent_state(4, np.array([[0, 0], [1, 1]]), 2)
    wall_loc, reward_location, agent_state = initialize_arena(4, 10)
    print(wall_loc.shape)
    print(reward_location.shape)
    print(agent_state.shape)