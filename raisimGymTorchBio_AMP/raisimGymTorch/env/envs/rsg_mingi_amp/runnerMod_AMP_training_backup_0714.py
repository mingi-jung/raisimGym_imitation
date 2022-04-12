from ruamel.yaml import YAML, dump, RoundTripDumper

from raisimGymTorch.env.bin import rsg_mingi_amp
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver

import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.algo.GAN.LSGAN as lsGAN

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


# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = os.path.abspath(task_path + "/../../../..")

# config
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')

# train config info
parser.add_argument('--cfgfile_name',    help='cfgfile name (str)',  type=str,     default=os.path.abspath(task_path + "/cfg.yaml"))
parser.add_argument('--num_envs',        help='num envs (int)', type=int,          default=-1)
parser.add_argument('--num_threads',     help='num threads (int)', type=int,       default=-1)
parser.add_argument('--eval_every_n',    help='eval_every_n (int)', type=int,      default=-1)
parser.add_argument('--simulation_dt',   help='simulation_dt (float)', type=float, default=-1)
parser.add_argument('--control_dt',      help='control_dt (float)', type=float,    default=-1)
parser.add_argument('--max_time',        help='max_time (float)', type=float,      default=-1)
parser.add_argument('--learning_rate',   help='learning rate (float)', type=float, default=-1)
parser.add_argument('--total_timesteps', help='total_timesteps (int)', type=int,   default=-1)
parser.add_argument('--policy_net',      help='policy_net (int)', type=int,        default=-1)
parser.add_argument('--value_net',       help='value_net (int)', type=int,         default=-1)
args = parser.parse_args()

cfg = YAML().load(open(args.cfgfile_name, 'r'))
if args.num_envs == -1:        args.num_envs = cfg['environment']['num_envs']
if args.num_threads == -1:     args.num_threads = cfg['environment']['num_threads']
if args.eval_every_n == -1:    args.eval_every_n = cfg['environment']['eval_every_n']
if args.simulation_dt == -1:   args.simulation_dt = cfg['environment']['simulation_dt']
if args.control_dt == -1:      args.control_dt = cfg['environment']['control_dt']
if args.max_time == -1:        args.max_time = cfg['environment']['max_time']
if args.learning_rate == -1:   args.learning_rate = cfg['environment']['learning_rate']
if args.total_timesteps == -1: args.total_timesteps = cfg['environment']['total_timesteps']
if args.policy_net == -1:      args.policy_net = cfg['architecture']['policy_net']
if args.value_net == -1:       args.value_net = cfg['architecture']['value_net']

mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda')

print(mode, "mode")
cfg['environment']['num_envs']          = args.num_envs
cfg['environment']['num_threads']       = args.num_threads
cfg['environment']['eval_every_n']      = args.eval_every_n
cfg['environment']['simulation_dt']     = args.simulation_dt
cfg['environment']['control_dt']        = args.control_dt
cfg['environment']['max_time']          = args.max_time
cfg['environment']['learning_rate']     = args.learning_rate
cfg['environment']['total_timesteps']   = args.total_timesteps
cfg['architecture']['policy_net']       = args.policy_net
cfg['architecture']['value_net']        = args.value_net

# create environment from the configuration file
env = VecEnv(impl=rsg_mingi_amp.RaisimGymEnv(task_path + "/rsc",
                                     dump(cfg['environment'],
                                     Dumper=RoundTripDumper)),
             cfg=cfg['environment'],
             normalize_ob=True,
             seed=0,
             normalize_rew=True,  # not implemented
             clip_obs=10.)

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
n_steps_per_update = n_steps * env.num_envs
n_update = int(cfg['environment']['total_timesteps'] / n_steps_per_update)
print("steps/update : ", n_steps_per_update, ", n_update : ", n_update)

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'],
                                        nn.LeakyReLU,
                                        ob_dim,
                                        act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'],
                                          nn.LeakyReLU,
                                          ob_dim,
                                          1),
                           device)

if mode == 'retrain':
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the policy:", weight_path+".pt\n")

    # save the configuration and related files to pre-trained model
    full_checkpoint_path = weight_path.rsplit(os.sep, 1)[0] + os.sep + 'full_' + weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1] + '.pt'
    mean_csv_path = weight_path.rsplit(os.sep, 1)[0] + os.sep + 'mean' + weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1] + '.csv'
    var_csv_path = weight_path.rsplit(os.sep, 1)[0] + os.sep + 'var' + weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1] + '.csv'
    saver = ConfigurationSaver(log_dir=os.path.abspath(home_path + "/data/runnerlog"),
                               save_items=[os.path.abspath(task_path + "/cfg.yaml"), os.path.abspath(task_path + "/Environment.hpp")],
                               pretrained_items=[weight_path.rsplit(os.sep, 1)[0].rsplit(os.sep, 1)[1],
                                                 [weight_path+'.pt', weight_path+'.txt', full_checkpoint_path, mean_csv_path, var_csv_path]])

    ## load observation scaling from files of pre-trained model
    env.load_scaling(weight_path.rsplit(os.sep, 1)[0], int(weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1]))
    print("Load observation scaling in", weight_path.rsplit(os.sep, 1)[0]+":", "mean"+str(int(weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1])) + ".csv", "and", "var"+str(int(weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1])) + ".csv")

    ## load actor and critic parameters from full checkpoint
    checkpoint = torch.load(full_checkpoint_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
else:
    # save the configuration and other files
    saver = ConfigurationSaver(log_dir=os.path.abspath(home_path + "/data/runnerlog"),
                               save_items=[os.path.abspath(task_path + "/cfg.yaml"), os.path.abspath(task_path + "/Environment.hpp")])

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              num_mini_batches=4,
              clip_param=0.2,
              gamma=0.998,
              lam=0.95,
              value_loss_coef=0.5,
              entropy_coef=0.0,
              learning_rate=cfg['environment']['learning_rate'],
              max_grad_norm=0.5,
              use_clipped_value_loss=True,
              log_dir=saver.data_dir,
              device=device,
              mini_batch_sampling='in_order',
              log_intervals=10)



## AMP discriminator

# D = lsGAN.LS_D()
D = torch.nn.Sequential(
    torch.nn.Linear(142, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1),
)
D_solver = optim.Adam(D.parameters(), lr=0.00005)
D_loss = 0


## AMP json load
with open('agent_data_normal.json') as f:
    json_data1 = json.load(f)

expert_data1 = json_data1['agent']
expert_data1 = torch.tensor(expert_data1)

with open('agent_data_fast.json') as f:
    json_data2 = json.load(f)

expert_data2 = json_data2['agent']
expert_data2 = torch.tensor(expert_data2)

with open('agent_data_slow.json') as f:
    json_data3 = json.load(f)

expert_data3 = json_data3['agent']
expert_data3 = torch.tensor(expert_data3)

with open('agent_data_run.json') as f:
    json_data4 = json.load(f)

expert_data4 = json_data4['agent']
expert_data4 = torch.tensor(expert_data4)




# expert_data = expert_data1 + expert_data2
expert_data = torch.cat([expert_data1,expert_data2,expert_data3,expert_data4], dim=0)

# print(len(expert_data))
# print(expert_data[1][0])
# print(expert_data[10001][0])
# print(expert_data[20001][0])
# print(expert_data[30001][0])

## AMP practice

# print("Model's state_dict:")
# for param_tensor in D.state_dict():
#     print(param_tensor, "\t", D.state_dict()[param_tensor].size())

# z = Variable(torch.randn(4, 142))
# D_real = D(z)
# D_loss = 0.5 * torch.mean((D_real - 1)**2)
# D_loss.backward()
# D_solver.step()
# D.zero_grad()

# print(z)
# print(D_real)
# print(D_loss)


# obs_amp = env.observe_amp()
#
# print(obs_amp)

# obs = env.observe()
# action = ppo.observe(obs)
# rewards, dones, infos = env.step(action) #infos[ienv]['episode'] has 'reward' and 'length' if it is done
# obs_amp = env.observe_amp()
# obs_amp = torch.tensor(obs_amp)
# # print(obs_amp) # 200 tensor
# N = 40000
# k = 200
# indices = torch.tensor(random.sample(range(N), k))
# indices = torch.tensor(indices)
# # expert_data = torch.tensor(expert_data)
# sampled_expert_data = expert_data[indices]
#
# D_real = D(sampled_expert_data)
# D_fake = D(obs_amp)
#
# print(obs_amp)
# # print(D_real)
# print(D_fake) # 200 tensor


# N = 10654
# k = 2
# # print(values)
# indices = torch.tensor(random.sample(range(N), k))
# indices = torch.tensor(indices)
# # print(indices)
# expert_data = torch.tensor(expert_data)
# sampled_expert_data = expert_data[indices]

# print(sampled_expert_data)

# print(z)
# print(D_real)
# print(D_loss)

# obs_amp = env.observe_amp()
# obs_amp = torch.tensor(obs_amp)



if mode == 'retrain':
    ## load optimizer parameters from full checkpoint
    ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

extraInfoNames = list()
avg_rewards = []

for update in range(n_update):
    start = time.time()

    # reset all environments
    # env.reset()   # by skoo because we will use the extrainfo
    env.reset_and_update_info()

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")

        # save the trained policy net and env scaling coefficients
        actor.save_deterministic_graph(
            saver.data_dir+"/policy_"+str(update)+'.pt',  # file name
            torch.rand(1, ob_dim).cpu())   # random observation to make the policy graph in torch.jit

        torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict()},
            os.path.abspath(saver.data_dir+"/full_"+str(update)+'.pt'))

        # save the actor parameter in a text file, too
        # parameters = np.zeros([0], dtype=np.float32)
        # for param in actor.deterministic_parameters():
        #     parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)
        # np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)

        # load the saved data to test it
        loaded_graph = torch.jit.load(os.path.abspath(saver.data_dir+"/policy_"+str(update)+'.pt'))

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        infos = list()
        for step in range(n_steps*2):
            time.sleep(0.01)
            obs = env.observe(update_mean=False)
            action_deterministic = loaded_graph(torch.from_numpy(obs).cpu())
            _, _, infos = env.step(action_deterministic.cpu().detach().numpy())

        env.stop_video_recording()
        env.turn_off_visualization()

        #env.reset()   # by skoo because we will use the extrainfo
        env.reset_and_update_info()

        # model.save(saver.data_dir+"/policies/policy", update)
        env.save_scaling(saver.data_dir, str(update))

        # SKOO to get extraInfoNames here ...
        if not extraInfoNames:  # if the list is empty
            infos_keys = list(infos[0].keys())
            for ikey in range(len(infos_keys)):
                if (infos_keys[ikey] != 'episode') and (infos_keys[ikey] != 'extrainfo'):
                    extraInfoNames.append(infos_keys[ikey])

    # actual training
    reward_sum = 0.
    done_sum = 0.
    average_dones = 0.

    count_done = 0.
    sum_length = 0.
    count_pass = 0.
    sum_length_pass = 0.

    full_episode_survival_check = np.ones(env.num_envs, dtype=int)
    rewards_ep = [[] for _ in range(env.num_envs)]
    extrainfos_ep = [[[] for _ in range(len(extraInfoNames))] for _ in range(env.num_envs)]




    # time_D_update_start = time.time()
    #
    # ## AMP discriminator update
    # obs = env.observe()
    # action = ppo.observe(obs)
    # rewards, dones, infos = env.step(action) #infos[ienv]['episode'] has 'reward' and 'length' if it is done
    # obs_amp = env.observe_amp()
    # obs_amp = torch.tensor(obs_amp)
    # # print(obs_amp) # 200 tensor
    # N = 40000
    # k = 200
    # indices = torch.tensor(random.sample(range(N), k))
    # indices = torch.tensor(indices)
    # # expert_data = torch.tensor(expert_data)
    # sampled_expert_data = expert_data[indices]
    #
    # D_real = D(sampled_expert_data)
    # D_fake = D(obs_amp)
    #
    # # print(D_real)
    # # print(D_fake) # 200 tensor
    #
    #
    # if update%40 == 0:
    #     D_loss = (torch.mean((D_real - 1)**2) + torch.mean((D_fake + 1)**2))
    #     D_loss.backward()
    #     D_solver.step()
    #     D.zero_grad()




    # print(D_loss)

    time_step_start = time.time()

    for step in range(n_steps):
        obs = env.observe()
        action = ppo.observe(obs)
        rewards, dones, infos = env.step(action) #infos[ienv]['episode'] has 'reward' and 'length' if it is done


        # ## AMP discriminator update
        # obs_amp = env.observe_amp()
        # obs_amp = torch.tensor(obs_amp)
        # # print(obs_amp) # 200 tensor
        # N = 10654
        # k = 200
        # indices = torch.tensor(random.sample(range(N), k))
        # indices = torch.tensor(indices)
        # expert_data = torch.tensor(expert_data)
        # sampled_expert_data = expert_data[indices]
        #
        # D_real = D(sampled_expert_data)
        # D_fake = D(obs_amp)
        #
        # # print(D_real)
        # # print(D_fake) # 200 tensor
        #
        # D_loss = (torch.mean((D_real - 1)**2) + torch.mean((D_fake + 1)**2))
        # D_loss.backward()
        # D_solver.step()
        # D.zero_grad()
        #
        # # print(D_loss)



        for ienv in range(env.num_envs):
            rewards_ep[ienv].append(rewards[ienv])
            for j in range(len(extraInfoNames)):
                extrainfos_ep[ienv][j].append(infos[ienv][extraInfoNames[j]])

        for ienv in range(env.num_envs):
            if dones[ienv] == 1:
                full_episode_survival_check[ienv] = 0
                info1 = infos[ienv]['episode']
                info1['reward_per_step'] = info1['reward'] / info1['length']  # add a new episode info
                infos[ienv]['episode'] = info1

                # eprew = sum(rewards_ep[ienv])
                eplen = len(rewards_ep[ienv])
                # epsteprew = eprew/eplen
                if info1['length'] != eplen:
                    print('abnormal episode length {}, {}'.format(eplen, info1['length']))

                count_done += 1
                sum_length += info1['length']
                rewards_ep[ienv].clear()

        if step == n_steps-1:
            for ienv in range(env.num_envs):
                if full_episode_survival_check[ienv] == 1:
                    eprew = sum(rewards_ep[ienv])
                    eplen = len(rewards_ep[ienv])
                    epsteprew = eprew / eplen
                    epinfo = {"reward": eprew, "length": eplen, "reward_per_step": epsteprew}
                    infos[ienv]['episode'] = epinfo   # update infos

                    count_pass += 1
                    sum_length_pass += eplen
                    rewards_ep[ienv].clear()

                extrainfo = dict()
                for j in range(len(extraInfoNames)):
                    extrainfo[extraInfoNames[j]] = sum(extrainfos_ep[ienv][j])/n_steps
                infos[ienv]['extrainfo'] = extrainfo   # update infos


                dloss = D_loss
                ampinfo = {"dloss": dloss}
                infos[ienv]['D_loss'] = ampinfo


        # collect roll-out information per step for ppo update
        # print("1")
        # print(rewards)

        ## AMP style reward append
        time_reward_start = time.time()
        obs_amp = env.observe_amp()
        obs_amp = torch.tensor(obs_amp)
        for ienv in range(env.num_envs):
            reward_style = 0.5 * max(0, 1 - 0.25 * (D(obs_amp[ienv]) - 1)**2)
            rewards[ienv] = rewards[ienv] + reward_style

        time_reward_end = time.time()

        ppo.step(value_obs=obs, rews=rewards, dones=dones, infos=infos)

        done_sum = done_sum + sum(dones)
        reward_sum = reward_sum + sum(rewards)

        # print(rewards)

    time_step_end_update_start = time.time()

    time_D_update_start = time.time()

    ## AMP discriminator update
    obs = env.observe()
    action = ppo.observe(obs)
    rewards, dones, infos = env.step(action) #infos[ienv]['episode'] has 'reward' and 'length' if it is done
    obs_amp = env.observe_amp()
    obs_amp = torch.tensor(obs_amp)

    # print(obs_amp[0]) # 200 tensor

    N = 40000
    k = 200
    indices = torch.tensor(random.sample(range(N), k))
    indices = torch.tensor(indices)
    # expert_data = torch.tensor(expert_data)
    sampled_expert_data = expert_data[indices]

    D_real = D(sampled_expert_data)
    D_fake = D(obs_amp)

    # print(D_real)
    # D_real = (D_real - 1)**2
    # print(D_real)
    # print(D_fake) # 200 tensor


    if update%50 == 0:
        D_loss = (torch.mean((D_real - 1)**2) + torch.mean((D_fake + 1)**2))
        D_loss.backward()
        D_solver.step()
        D.zero_grad()

        torch.save(D.state_dict(), os.path.abspath(saver.data_dir+"/AMP_D_"+str(update)+'.pt'))

    # print(D_loss)

    # take st step to get value obs
    obs = env.observe()
    ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)


    end = time.time()

    average_performance = reward_sum / n_steps_per_update
    average_dones = done_sum / n_steps_per_update
    avg_rewards.append(average_performance)

    actor.distribution.enforce_minimum_std((torch.ones(25)*0.2).to(device))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average per step reward: ", '{:0.10f}'.format(average_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(n_steps_per_update / (end - start))))

    print('{:<40} {:>6}'.format("time elapsed in D update: ", '{:6.4f}'.format(time_step_start - time_D_update_start)))
    print('{:<40} {:>6}'.format("time elapsed in step loop: ", '{:6.4f}'.format(time_step_end_update_start - time_step_start)))
    print('{:<40} {:>6}'.format("time elapsed in reward calculation: ", '{:6.4f}'.format(time_reward_end - time_reward_start)))
    print('{:<40} {:>6}'.format("time elapsed in ppo update: ", '{:6.4f}'.format(end - time_step_end_update_start)))

    print('{:<40} {:>6}'.format("Discriminator real: ", '{:6.4f}'.format(torch.mean(D_real))))
    print('{:<40} {:>6}'.format("Discriminator fake: ", '{:6.4f}'.format(torch.mean(D_fake))))
    print('{:<40} {:>6}'.format("Discriminator loss: ", '{:6.4f}'.format(D_loss)))

    print('{:<40} {:>6}'.format("count done: ", '{:6.0f}'.format(count_done)))
    if count_done > 0:
        print('{:<40} {:>6}'.format("average done length: ", '{:6.0f}'.format(sum_length / count_done)))
    print('{:<40} {:>6}'.format("count pass: ", '{:6.0f}'.format(count_pass)))
    if count_pass > 0:
        print('{:<40} {:>6}'.format("average pass length: ", '{:6.0f}'.format(sum_length_pass / count_pass)))

    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
