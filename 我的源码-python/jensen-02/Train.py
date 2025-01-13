import gymnasium as gym
import numpy as np
import torch
from Scripts.Setting.Setting import set_seed, LossHP
from Scripts.MazeGameEnv import MazeGameEnv
from Scripts.Agent import Agent, ModelProperties, Model
from Scripts.Planner import Planner
from torch.optim import Adam
import copy

torch.autograd.set_detect_anomaly(True)

"""
        "--Nhidden"
        help = "Number of hidden units"
        arg_type = Int
        default = 100
        "--Larena"
        help = "Arena size (per side)"
        arg_type = Int
        default = 4
        "--T"
        help = "Number of timesteps per episode (in units of physical actions)"
        arg_type = Int
        default = 50
        "--Lplan"
        help = "Maximum planning horizon"
        arg_type = Int
        default = 8
        "--load"
        help = "Load previous model instead of initializing new model"
        arg_type = Bool
        default = false
        "--load_epoch"
        help = "which epoch to load"
        arg_type = Int
        default = 0
        "--seed"
        help = "Which random seed to use"
        arg_type = Int
        default = 1
        "--save_dir"
        help = "Save directory"
        arg_type = String
        default = "./"
        "--beta_p"
        help = "Relative importance of predictive loss"
        arg_type = Float32
        default = 0.5f0
        "--prefix"
        help = "Add prefix to model name"
        arg_type = String
        default = ""
        "--load_fname"
        help = "Model to load (default to default model name)"
        arg_type = String
        default = ""
        "--n_epochs"
        help = "Total number of training epochs"
        arg_type = Int
        default = 1001
        "--batch_size"
        help = "Batch size for each gradient step"
        arg_type = Int
        default = 1 # 40
        "--lrate"
        help = "Learning rate"
        arg_type = Float64
        default = 1e-3
        "--constant_rollout_time"
        help = "Do rollouts take a fixed amount of time irrespective of length"
        arg_type = Bool
        default = true
"""

args: dict = {
    "Nhidden": 100,
    "Larena": 4,
    "T": 50,
    "Lplan": 8,
    "load": False,
    "load_epoch": 0,
    "seed": 1,
    "save_dir": "./Output",
    "beta_p": 0.5,
    "prefix": "",
    "load_fname": "",
    "n_epochs": 1001,
    "batch_size": 5,
    "lrate": 1e-3,
    "constant_rollout_time": True
}

def main():
    # 设置随机种子
    random_seed = args["seed"]
    set_seed(random_seed)
    
    # 参数
    Larena = args["Larena"]
    Lplan = args["Lplan"]
    Nhidden = args["Nhidden"]
    T = args["T"]
    load = args["load"]
    seed = args["seed"]
    save_dir = args["save_dir"]
    prefix = args["prefix"]
    load_fname = args["load_fname"]
    n_epochs = int(args["n_epochs"])
    βp = float(args["beta_p"])
    batch_size = int(args["batch_size"])
    lrate = int(args["lrate"])
    constant_rollout_time = int(args["constant_rollout_time"])
    
    action_size = 5
    wrap = True
    
    planner = Planner(plan_depth=Lplan, maze_size=Larena, batch_size=batch_size)
    state_size = Larena * Larena
    
    output_size = action_size + 1 + state_size #actions and value function and prediction of state
    # todo 不知道干啥用的
    # output_size += 1 # needed for backward compatibility (this lives between state and reward predictions)
    wall_input_size = 2 * state_size
    input_size = action_size + 1 + 1 + state_size + wall_input_size
    input_size += planner.plan_in_size
    output_size += planner.plan_out_size

    loss_hp = LossHP(
        # predictive loss weight
        βp=βp,
        # value function loss weight
        βv=0.05,
        # entropy loss cost
        βe=0.05,
        # reward loss cost
        βr=1.0,
    )
    
    # 注册 并初始化环境
    gym.register(
        id="gymnasium_env/MazeGameEnv-v0",
        entry_point=MazeGameEnv,
        kwargs={
            "batch_size": batch_size,
            "maze_size": Larena,
            "wrap": wrap,
        }
    )
    
    env = gym.make("gymnasium_env/MazeGameEnv-v0")
    
    obs, info = env.reset(seed=2, options=None)
    
    mp = ModelProperties(
        device=  torch.device("cuda" if torch.cuda.is_available() else "cpu"),#torch.device("cuda" if torch.cuda.is_available() else "cpu"),torch.device("cpu"),
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=Nhidden,
        maze_size=Larena,
        action_size=action_size,
        out_size=output_size,
        is_greedy=False
    )
    
    model = Model(mp)
    
    model.to(model.properties.device)
    
    optimizer = Adam(model.parameters(), lr=lrate)
    
    agent = Agent(
        model=model,
        optimizer=optimizer,
        planner=planner,
        device=mp.device,
    )
    
    for epoch in range(n_epochs):
        epoch_end = False
        
        epoch_rew = torch.tensor([], dtype=torch.float32, device=mp.device)
        epoch_model_output = torch.tensor([], dtype=torch.float32, device=mp.device)
        epoch_action = torch.tensor([], dtype=torch.int32, device=mp.device)
        epoch_active = torch.tensor([], dtype=torch.int32, device=mp.device)
        
        
        L = torch.tensor(0, dtype=torch.float32, device=mp.device)
        Lprior = torch.tensor(0, dtype=torch.float32, device=mp.device)
        Lpred = torch.tensor(0, dtype=torch.float32, device=mp.device)
        
        active = np.ones((batch_size), dtype=np.int32)
        
        sum_rew = 0
        
        while not epoch_end:
            model_output, a = agent.gen_action(obs=obs)
            Lprior += agent.prior_loss(model_output, active, a)
            
            obs, reward, active, epoch_end, info = env.step(a.numpy())
            
            sum_rew += np.sum(reward)
            
            agent_true_idx, reward_true_idx = agent.get_agent_reward_loc_index_from_obs(obs=obs)
            
            Lpred += agent.calc_agent_prediction_loss(model_output, agent_true_idx, active)
            Lpred += agent.calc_reward_prediction_loss(model_output, reward_true_idx, active)
            
            epoch_rew = torch.cat((epoch_rew, torch.tensor(reward).unsqueeze(0).to(mp.device)))
            epoch_model_output = torch.cat((epoch_model_output, model_output.unsqueeze(0).to(mp.device)))
            epoch_action = torch.cat((epoch_action, a.unsqueeze(0).to(mp.device)))
            epoch_active = torch.cat((epoch_active, torch.tensor(active).unsqueeze(0).to(mp.device)))
            
            # epoch_end= True
        
        TD_error = agent.calc_deltas(epoch_model_output[:, :, 0, action_size], epoch_rew)
        
        N = epoch_rew.shape[0]
        
        for t in range(N):
            active = epoch_active[t]
            Vterm = TD_error[t, :] * epoch_model_output[t, :, 0, action_size]
            L -= torch.sum(loss_hp.βv * Vterm[active])
            for b in range(batch_size):
                if active[b]:
                    RPE_term = TD_error[t, b] * epoch_model_output[t, b, 0, epoch_action[t, b]] #PG term
                    L -= loss_hp.βr * RPE_term #add to loss

        L += (loss_hp.βp * Lpred) #add predictive loss for internal world model
        L -= loss_hp.βe * Lprior #add prior loss (formulated as a likelihood above)
        L /= batch_size #normalize by batch

        if (epoch - 1) % 10 == 0:
            print(f"epoch = {epoch}, L = {L}, rew = {sum_rew / batch_size}")
        
        agent.backward(L)
        agent.update()
        agent.clear_grad()
        agent.planner.clear_plan_cache()
        agent.clear_h()
        

        obs, info = env.reset()
    
    # print(obs["walls_loc"][2, :, :])

    
if __name__ == '__main__':
    main()