from ruamel.yaml import YAML, dump, RoundTripDumper

from raisimGymTorch.env.bin import rsg_gaitmsk_adaptation as rsg_anymal
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

import class_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

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
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    classifier = class_classification.classify_net(142, device=device, learning_rate=0.001)
    classifier.load_state("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/classification/data/full_classifier_9500.pt")

    total = 0
    correct = 0

    # max_steps = 1000000
    max_steps = 12000 ## 10 secs
    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        obs_amp_0 = env.observe_amp()
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward_ll, dones, infos = env.step(action_ll.cpu().detach().numpy())
        obs_amp_1 = env.observe_amp()
        reward_ll_sum = reward_ll_sum + reward_ll[0]

        predicted = classifier.test(obs_amp_0, obs_amp_1)
        # print(predicted)
        total += 1
        if predicted == 1: ##subject number check needs
            correct += 1
        print(f'Accuracy: {100 * correct // total} %')


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
