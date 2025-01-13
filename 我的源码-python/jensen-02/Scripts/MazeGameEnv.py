from typing import Optional
import numpy as np
import gymnasium as gym
import copy
from Scripts.Initializations import initialize_arena, reset_agent_state

class MazeGameEnv(gym.Env):
    def __init__(self, batch_size: int, maze_size: int, wrap: bool = True):
        super(MazeGameEnv, self).__init__()
        self.batch_size = batch_size
        self.maze_size = maze_size
        self.wrap = wrap
        
        self.reset(seed=None, options=None)
        
        # 观测空间
        self.observation_space = gym.spaces.Dict(
            {
                "act": gym.spaces.Box(low=0, high=4, shape=(batch_size, ), dtype=np.int32),
                "reward": gym.spaces.Box(low=0, high=2, shape=(batch_size, ), dtype=np.float32),
                "time": gym.spaces.Box(low=0, high=20000, shape=(batch_size, ), dtype=np.int32),
                "agent_loc": gym.spaces.Box(low=0, high=self.maze_size - 1, shape=(batch_size, 2,), dtype=np.int32),
                "reward_loc": gym.spaces.Box(low=0, high=self.maze_size - 1, shape=(batch_size, 2,), dtype=np.int32),
                "walls_loc": gym.spaces.Box(low=0, high=1, shape=(batch_size, self.maze_size ** 2, 4), dtype=np.int32),  # 墙的位置信息
            }
        )

        # We have 5 actions, corresponding to "up", "right",  "down", "left", "stay"
        self.action_space = gym.spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array([-1, 0]),       # up
            1: np.array([0, 1]),        # right
            2: np.array([1, 0]),        # down
            3: np.array([0, -1]),       # left
            4: np.array([0, 0]),        # stay
        }
        
        self.action_spend_time = {
            0: 400,         # up
            1: 400,         # right
            2: 400,         # down
            3: 400,         # left
            4: 120,         # stay
        }
    
    def reset(self, seed: Optional[int] = None, options: dict[str, any] | None = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        wall_loc, reward_location, agent_state = initialize_arena(Larena=self.maze_size, batch=self.batch_size)
        self.wall_loc = wall_loc.astype(np.int32)
        self.reward_location = reward_location.astype(np.int32)
        self.agent_state = agent_state.astype(np.int32)
        
        self.time = np.zeros((self.batch_size, ), dtype=np.int32)
        self.active = np.ones((self.batch_size, ), dtype=np.int32)
        
        observation = self._get_obs(act=np.zeros((self.batch_size, ), dtype=np.int32), reward=np.zeros((self.batch_size, ), dtype=np.float32))
        info = self._get_info()

        return observation, info
    
    # return
    # obs, reward, active, episode_end, info
    def step(self, action: np.ndarray):
        # Get the reward
        reward = self.take_action(action=action)
        
        # Get the observation
        observation = self._get_obs(act=action, reward=reward)
        info = self._get_info()
        
        # 只有当self.active全为0是episode_end才为True
        episode_end = np.all(self.active == 0)
        
        return observation, reward, self.active.astype(np.bool_), episode_end, info

    def _get_obs(self, act: Optional[np.ndarray] = None, reward: Optional[np.ndarray] = None):
        obs = {
            "act": act,
            "reward": reward,
            "time": self.time,
            "agent_loc": self.agent_state,
            "reward_loc": self.reward_location,
            "walls_loc": self.wall_loc,
        }
        # print(obs)
        return copy.deepcopy(obs)

    def _get_info(self, ):
        return {}

    def take_action(self, action: np.ndarray):
        # Check if the action is valid
        assert action.shape == (self.batch_size, )
        assert np.all(0 <= action) and np.all(action < 5)
        
        # Get the current state
        agent_state = self.agent_state
        reward_location = self.reward_location
        wall_loc = self.wall_loc
        time = self.time
        active = self.active
        batch_size = self.batch_size
        
        reward = np.zeros((batch_size, ), dtype=np.float32)
        
        # Get the next state
        for b in range(batch_size):
            if active[b] == 0:
                continue
            
            time[b] += self.action_spend_time[action[b]]
            if action[b] != 4:
                new_loc = agent_state[b] + self.action_to_direction[action[b]]
                
                if self.wrap:
                    new_loc = new_loc % self.maze_size
                
                # 判断是否撞墙
                if 0 <= new_loc[0] < self.maze_size and 0 <= new_loc[1] < self.maze_size:
                    if wall_loc[b, new_loc[0] * self.maze_size + new_loc[1], action[b]] == 0:
                        agent_state[b] = new_loc
        
            # 判断是否到达奖励位置
            if np.array_equal(agent_state[b], reward_location[b]):
                reward[b] = 1
            
            # 每个batch只在剩余时间不足以行动之后才会失活
            if time[b] > 20000 - 400:
                active[b] = 0
            
            # 如果到达奖励位置，且仍然活跃，则重置agent_state
            if reward[b] == 1 and active[b] == 1:
                # reset agent state
                agent_state[b] = reset_agent_state(Larena=self.maze_size, reward_location=self.reward_location, batch=1)
        self.agent_state = agent_state
        return reward