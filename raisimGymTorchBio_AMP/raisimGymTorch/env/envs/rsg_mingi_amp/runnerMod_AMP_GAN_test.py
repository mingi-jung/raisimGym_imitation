from ruamel.yaml import YAML, dump, RoundTripDumper

from raisimGymTorch.env.bin import rsg_mingi_amp as rsg_anymal
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver

import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO

import os
import math
import time
import datetime
import argparse

import torch
import torch.nn as nn
import numpy as np

## AMP
import torch.optim as optim
import torch.autograd as autograd
import json
from torch.autograd import Variable
import random


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = os.path.abspath(task_path + "/../../../..")

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

env = VecEnv(rsg_anymal.RaisimGymEnv(task_path + "/rsc",
                                     dump(cfg['environment'],
                                     Dumper=RoundTripDumper)),
             cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts




## AMP discriminator

# D = lsGAN.LS_D()
D = torch.nn.Sequential(
    torch.nn.Linear(142, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1),
)
D_solver = optim.Adam(D.parameters(), lr=0.001)





## AMP json load
# with open('agent_data_mingi.json') as f:
#     json_data = json.load(f)
with open('AMP_subject06_01.json') as f:
    json_data = json.load(f)

# expert_data = json_data['agent']
expert_data = json_data['expert']
expert_data = torch.tensor(expert_data)

weight_path = args.weight
if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy", weight_path+".pt")
    loaded_graph = torch.jit.load(weight_path+'.pt')

    env.load_scaling(weight_path.rsplit(os.sep, 1)[0], int(weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1]))
    print("Load observation scaling in", weight_path.rsplit(os.sep, 1)[0]+":", "mean"+str(int(weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1])) + ".csv", "and", "var"+str(int(weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1])) + ".csv")
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 10000 ## 10 secs
    for step in range(max_steps):

        if step%1200 == 0:
            env.reset()

        time.sleep(0.01)
        obs = env.observe(False)
        # obs_amp = env.observe_amp(False) # amp mingi
        action_ll = loaded_graph(torch.from_numpy(obs).cpu())
        reward_ll, dones, infos = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]

        ## AMP discriminator update
        obs_amp = env.observe_amp()
        obs_amp = torch.tensor(obs_amp)
        # print(obs_amp) # 200 tensor
        N = 10654
        # N = 1500
        k = 200
        indices = torch.tensor(random.sample(range(N), k))
        indices = torch.tensor(indices)
        # expert_data = torch.tensor(expert_data)
        sampled_expert_data = expert_data[indices]

        D_real = D(sampled_expert_data)
        D_fake = D(obs_amp)

        # print(D_real)
        # print(D_fake) # 200 tensor

        D_loss = (torch.mean((D_real - 1)**2) + torch.mean((D_fake + 1)**2))
        D_loss.backward()
        D_solver.step()
        D.zero_grad()

        # if step%100 == 0:
        #     print('{:<40} {:>6}'.format("Discriminator real: ", '{:6.4f}'.format(torch.mean(D_real))))
        #     print('{:<40} {:>6}'.format("Discriminator fake: ", '{:6.4f}'.format(torch.mean(D_fake))))
        #     print('{:<40} {:>6}'.format("Discriminator loss: ", '{:6.4f}'.format(D_loss)))

        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("sum reward: ", '{:0.10f}'.format(reward_ll_sum)))
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
