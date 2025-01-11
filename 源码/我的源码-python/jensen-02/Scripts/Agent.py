import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam  # 导入优化器
from Scripts.Planner import Planner
import copy

class ModelProperties():
    def __init__(
        self, 
        device: torch.device,
        batch_size: int,
        input_size: int,
        hidden_size: int, 
        maze_size: int,
        action_size: int,
        out_size: int,
        is_greedy: bool = False,
        ) -> None:
        self.device = device
        self.state_size = maze_size ** 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maze_size = maze_size
        self.action_size = action_size
        self.out_size = out_size
        self.policy_out_size = action_size + 1
        self.pre_out_size = out_size - action_size - 1
        self.pre_in_size = hidden_size + action_size
        self.batch_size = batch_size
        self.is_greedy = is_greedy
        
class Model(nn.Module):
    def __init__(self, mp:ModelProperties) -> None:
        super(Model, self).__init__()
        self.properties = mp
        self._init_net()
        
    def _init_net(self):
        prop = self.properties
        
        # 定义GRU层
        self.gru = nn.GRU(input_size=prop.input_size, hidden_size=prop.hidden_size, batch_first=True)
        
        # 定义输出策略和价值函数的全连接层
        self.policy = nn.Linear(prop.hidden_size, prop.policy_out_size)  # +1 for value function
        
        # 定义输出预测的全连接层(也就是内部世界模型)
        self.prediction = nn.Sequential(
            nn.Linear(prop.pre_in_size, prop.pre_out_size),
            nn.ReLU(),
            nn.Linear(prop.pre_out_size, prop.pre_out_size),
        )
        
    def forward(self, x, h):
        # GRU层
        rnn_output, h = self.gru(x, h)
        
        # 输出策略和价值函数
        policy = self.policy(rnn_output)
        
        # 分离策略和价值函数
        logπ, V = policy[:, :, :-1], policy[:, :, -1]
        V = V.unsqueeze(-1)
        
        logπ = logπ - torch.logsumexp(logπ, dim=-1, keepdim=True) #softmax for normalization
        
        a = self.sample_actions(logπ)
        
        ahot = self.construct_ahot(a)
        
        prediction_input = torch.cat((rnn_output.to(self.properties.device), ahot.to(self.properties.device)), dim=-1)  # input to prediction module (concatenation of hidden state and action)
        prediction_output = self.prediction(prediction_input)  # output of prediction module
        
        return rnn_output, h, torch.cat((logπ, V, prediction_output), dim=-1), a
    
    # 由概率得到动作
    @torch.no_grad()
    def sample_actions(self, policy_logits):
        batch = policy_logits.shape[0]
        a = torch.zeros((batch, 1), dtype=torch.int32)

        πt = torch.exp(policy_logits)
        πt = πt / πt.sum(dim=-1, keepdim=True)
        
        if torch.isnan(πt).any() or torch.isinf(πt).any():
            for b in range(batch):
                if torch.isnan(πt[b, :]).any() or torch.isinf(πt[b, :]).any():
                    πt[b, :] = torch.zeros_like(πt[b, :])
                    πt[b, torch.argmax(policy_logits[b, :])] = 1
        
        if self.properties.is_greedy:
            a[:, 0] = torch.argmax(πt, dim=-1).int()
        else:
            a[:, 0] = torch.tensor([Categorical(πt[b, :]).sample().item() for b in range(batch)], dtype=torch.int32)

        return a
    
    @torch.no_grad()
    def construct_ahot(self, a):
        batch = a.shape[0]
        ahot = torch.zeros((batch, 1, self.properties.action_size), dtype=torch.float32, device=self.properties.device)
        for b in range(batch):
            ahot[b, 0, a[b, 0]] = 1.0
        return ahot
    
class Agent():
    def __init__(self, model:Model, planner:Planner, optimizer:Adam, device:torch.device):
        self.model = model
        self.planner = planner
        self.optimizer = optimizer
        self.device = device
        self.old_h = self.clear_h()
        
    def gen_action(self, obs):
        # print(f"obs = {obs}")
        model_input = self.gen_model_input(obs)
        rnn_ouput, new_h, model_output, a = self.model.forward(model_input, self.old_h)
        self.old_h = new_h
        self.rollout(obs, rnn_ouput, model_output, a)
        return model_output, self.trans_act_form(a)
    
    def rollout(self, obs, h, model_output, a):
        prop = self.model.properties
        maze_size = prop.maze_size
        action_size = prop.action_size
        state_size = prop.state_size
        batch_size = prop.batch_size
        plan_depth = self.planner.plan_depth
        
        plan_path = torch.zeros((batch_size, plan_depth, 4), dtype=torch.int32)
        # all_Vs = torch.zeros((batch_size, ), dtype=torch.float32)
        found_rew = torch.zeros((batch_size, ), dtype=torch.float32)
        # plan_states = torch.zeros((batch_size, plan_depth), dtype=torch.int32)

        
        # 从model_output中，获取预测的奖励位置
        goals = self.get_goals_from_model_output(model_output)
        # 获取action为4的索引
        plan_inds = torch.where(a == 4)[0].to(self.device)
        
        new_h = h.permute(1, 0, 2)
        h_rnn = new_h[:, plan_inds, :]
        # 交换h_rnn的第一和第二维度
        rnn_output = h_rnn.permute(1, 0, 2).to(self.device)
        goal = goals[plan_inds, :, :].to(self.device)
        
        walls_input = self.gen_wall_input_from_obs_wall(obs["walls_loc"]).to(self.device)
        states_input = self.gen_state_input_from_obs_state(obs["agent_loc"]).to(self.device)
        times_input = self.gen_time_input_from_obs_time(obs["time"]).to(self.device)
        
        wall_input = walls_input[plan_inds, :].to(self.device)
        state_input = states_input[plan_inds, :].to(self.device)
        time_input = times_input[plan_inds, :].to(self.device)
        
        batch = len(plan_inds)
        model_input = torch.zeros((batch, 1, prop.input_size), dtype=torch.float32, device=self.device)
        
        for n_steps in range(plan_depth):
            if n_steps > 0:  
                rnn_output, h_rnn = self.model.gru(model_input, h_rnn)
            
            policy = self.model.policy(rnn_output)
            
            logπ, V = policy[:, :, :-2], policy[:, :, -1]
            V = V.unsqueeze(-1)
            
            logπ = logπ - torch.logsumexp(logπ, dim=-1, keepdim=True)
            
            act = self.model.sample_actions(logπ)
            # 记录行动
            for (index, b) in enumerate(plan_inds):
                plan_path[b, n_steps, act[index]] = 1.0
            
            act_input = self.model.construct_ahot(act).to(self.device)
            
            prediction_input = torch.cat((rnn_output.to(self.device), act_input.to(self.device)), dim=-1)
            prediction_output = self.model.prediction(prediction_input)
            
            # 更新agent的位置
            new_states = prediction_output[:, :, :state_size]
            new_states = new_states - torch.logsumexp(new_states, dim = -1, keepdim=True)
            state_dist = torch.exp(new_states)
            
            # 找到每个张量中最大值的索引, 创建one-hot编码
            new_states_max_idx = torch.argmax(state_dist, dim=-1)
            new_states_one_hot = torch.zeros_like(new_states)
            new_states_one_hot.scatter_(2, new_states_max_idx.unsqueeze(-1), 1)
            
            # 检查是否到达目标位置
            matches = torch.all(new_states_one_hot == goal, dim=-1)

            found_rew[plan_inds[matches[:, 0]]] = 1.0

            # not_finished = ~matches[:, 0]
            not_finished = torch.nonzero(~matches[:, 0]).squeeze(1)
            
            if len(not_finished) == 0:
                break
            
            ### only consider active states going forward ###
            h_rnn = h_rnn[:, not_finished, :]
            goal = goal[not_finished]
            plan_inds = plan_inds[not_finished]
            
            reward_input = torch.zeros((not_finished.shape[0], 1, 1), dtype=torch.float32)
            act_input = act_input[not_finished]
            time_input = time_input[not_finished] + 0.02 #increment time
            state_input = state_input[not_finished]
            wall_input = wall_input[not_finished]
            plan_input = self.planner.get_plan_cache().to(self.device)[plan_inds].to(self.device)
            
            model_input = torch.cat((act_input.to(self.device), reward_input.to(self.device), time_input.to(self.device), state_input.to(self.device), wall_input.to(self.device), plan_input.to(self.device)), dim=-1)
            
            
        self.planner.store_plan_cache(plan_path=plan_path, found_rew=found_rew)
        return plan_path, found_rew
    
    @torch.no_grad()
    def U_prior(self):
        action_size = self.model.properties.action_size
        batch_size = self.model.properties.batch_size
        return torch.ones((batch_size, 1, action_size)) / action_size
    
    def prior_loss(self, model_output, active, action):
        action_size = self.model.properties.action_size
        
        logp = torch.log(self.U_prior().to(self.device)).to(self.device)
        logπ = model_output[:, :, :action_size]
        
        logπ = logπ[active, :, :]
        logπ = logπ[:, :, action]
        
        logp = logp[active, :, :]
        logp = logp[:, :, action]
        
        lprior = torch.sum(torch.exp(logπ) * (logp - logπ))
        return lprior  # return the negative KL divergence as the loss
    
    def calc_agent_prediction_loss(self, model_output, agent_idx, active):
        Naction = self.model.properties.action_size
        Nstates = self.model.properties.state_size

        new_Lpred = 0.0 #initialize prediction loss
        spred = model_output[:, :, (Naction + 1):(Naction + 1 + Nstates)] #predicted next states (Nstates x batch)
        spred = spred - torch.logsumexp(spred, dim=-1, keepdim=True) #softmax over states

        # 使用矩阵运算new_Lpred += spred[b, 0, agent_idx[b]] if active[b] else 0
        # 使用矩阵运算计算预测损失
        active_indices = torch.nonzero(torch.tensor(active)).squeeze(1)
        new_Lpred = -torch.sum(spred[active_indices, 0, agent_idx[active_indices]])
        return new_Lpred #return summed loss
        
    def calc_reward_prediction_loss(self, model_output, reward_idx, active):
        Naction = self.model.properties.action_size
        Nstates = self.model.properties.state_size

        new_Lpred = 0.0 #initialize prediction loss
        spred = model_output[:, :, -Nstates:] #predicted next states (Nstates x batch)
        spred = spred - torch.logsumexp(spred, dim=-1, keepdim=True) #softmax over states
        # 使用矩阵运算计算预测损失
        active_indices = torch.nonzero(torch.tensor(active)).squeeze(1)
        new_Lpred = -torch.sum(spred[active_indices, 0, reward_idx[active_indices]])
        return new_Lpred #return summed loss
    
    @torch.no_grad()
    def calc_deltas(self, pre_rew, true_rew):
        N, batch_size = pre_rew.shape
        δs = torch.zeros((N, batch_size), dtype=torch.float32, device=self.device) #initialize TD errors
        R = torch.zeros((batch_size), dtype=torch.float32, device=self.device) #cumulative reward
        for t in range(N): #for each iteration (moving backward!)
            R = true_rew[N - t - 1, :] + R #cumulative reward
            δs[N - t - 1, :] = R - pre_rew[N - t - 1, :] #TD error
        return δs
    
    def clear_grad(self,):
        self.optimizer.zero_grad()  # 清空之前的梯度

    def backward(self, L):
        L.backward(retain_graph=True)  # 反向传播

    def update(self):
        self.optimizer.step()  # 更新参数
    
    def clear_h(self):
        self.old_h = torch.zeros((1, self.model.properties.batch_size, self.model.properties.hidden_size), dtype=torch.float32).to(self.device)
    
    def get_goals_from_model_output(self, model_output):
        prop = self.model.properties
        # 从 model_output 中获取goal 即：假定的目标位置
        maze_size = prop.maze_size
        pre_out_size = prop.pre_out_size
        state_size = prop.state_size
        
        prediction_output = model_output[:, :, -pre_out_size:]
        
        rew_states = prediction_output[:, :, state_size:]
        rew_states_sm = rew_states - torch.logsumexp(rew_states, dim=-1, keepdim=True)
        state_dist = torch.exp(rew_states_sm)
        
        rew_states_max_idx = torch.argmax(state_dist, dim=-1)
        rew_states_one_hot = torch.zeros_like(rew_states_sm)

        rew_states_one_hot[torch.arange(0, prop.batch_size), 0, rew_states_max_idx[:, 0]] = 1

        return rew_states_one_hot

    def get_agent_reward_loc_index_from_obs(self, obs):
        maze_size = self.model.properties.maze_size
        agent_loc = obs["agent_loc"]
        reward_loc = obs["reward_loc"]
        batch_size = agent_loc.shape[0]
        agent_idx = torch.zeros((batch_size), dtype=torch.int32)
        reward_idx = torch.zeros((batch_size), dtype=torch.int32)
        
        for b in range(batch_size):
            agent_idx[b] = agent_loc[b][0] * maze_size + agent_loc[b][1]
            reward_idx[b] = reward_loc[b][0] * maze_size + reward_loc[b][1]
        
        return agent_idx, reward_idx
        
    def trans_act_form(self, a):
        # torch.tensor 的a转化为 numpy 的a
        # 并且将a的维度从 (batch_size, 1) 转化为 (batch_size, )
        res = a.squeeze(1)
        return res

    def gen_model_input(self, obs):
        act_input = self.gen_act_input_from_obs_act(obs["act"])
        reward_input = self.gen_reward_input_from_obs_reward(obs["reward"])
        time_input = self.gen_time_input_from_obs_time(obs["time"])
        state_input = self.gen_state_input_from_obs_state(obs["agent_loc"])
        wall_input = self.gen_wall_input_from_obs_wall(obs["walls_loc"])
        plan_input = self.planner.get_plan_cache()
        # todo 后面可以优化，直接拼接
        model_input = torch.cat((act_input.to(self.device), reward_input.to(self.device), time_input.to(self.device), state_input.to(self.device), wall_input.to(self.device), plan_input.to(self.device)), dim=-1)
        model_input.to(self.device)
        return model_input
    
    def get_each_model_input_size(self):
        prop = self.model.properties
        act_size = prop.action_size
        state_size = prop.state_size
        wall_input_size = 2 * state_size
        plan_in_size = self.planner.plan_in_size
        return act_size, state_size, wall_input_size, plan_in_size
    
    def gen_act_input_from_obs_act(self, act):
        batch_size = self.model.properties.batch_size
        action_size = self.model.properties.action_size
        
        # 创建一个全零的张量
        act_input = torch.zeros((batch_size, 1, action_size), dtype=torch.float32)
        
        # 使用索引操作来设置值
        act_input[torch.arange(batch_size), 0, act] = 1.0
        
        return act_input
    
    def gen_reward_input_from_obs_reward(self, reward):
        reward_input = torch.zeros((self.model.properties.batch_size, 1, 1), dtype=torch.float32)
        reward_input[:, 0, 0] = torch.tensor(reward)
        return reward_input
    
    def gen_time_input_from_obs_time(self, time):
        time_input = torch.zeros((self.model.properties.batch_size, 1, 1), dtype=torch.float32)
        time_input[:, 0, 0] = torch.tensor(time)
        return time_input
    
    def gen_state_input_from_obs_state(self, state):
        state_input = torch.zeros((self.model.properties.batch_size, 1, self.model.properties.maze_size ** 2), dtype=torch.float32)
        state_input[:, 0, :] = self.from_2d_state_to_one_hot(state)
        return state_input
    
    def gen_wall_input_from_obs_wall(self, wall):
        batch_size = self.model.properties.batch_size
        state_size = self.model.properties.state_size
        
        # 创建一个全零的张量
        wall_input = torch.zeros((batch_size, 1, 2 * state_size), dtype=torch.float32)
        
        walls = torch.tensor(wall[:, :, :2])
        # 将 walls 的第 2 和第 3 维度拉平
        walls_flat = walls.view(batch_size, -1)
        
        # 将拉平后的 walls 赋值给 wall_input
        wall_input[:, 0, :walls_flat.shape[1]] = walls_flat
        
        return wall_input
        
    def from_2d_state_to_one_hot(self, state):
        batch_size = state.shape[0]
        maze_size = self.model.properties.maze_size
        state_one_hot = torch.zeros((batch_size, self.model.properties.maze_size ** 2), dtype=torch.float32)
        for b in range(batch_size):
            state_one_hot[b, state[b][0] * maze_size + state[b][1]] = 1.0
        return state_one_hot
    
