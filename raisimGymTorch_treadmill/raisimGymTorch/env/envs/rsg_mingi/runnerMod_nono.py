from ruamel.yaml import YAML, dump, RoundTripDumper

from raisimGymTorch.env.bin import rsg_mingi
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

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
parser = argparse.ArgumentParser()
parser.add_argument('--mode',                  help='mode', type=str, default='train')
# parser.add_argument('--study_name',            help='study name (str)', type=str, default='regular')
parser.add_argument('--test_modelfile_name',   help='modelfile name (str)',   type=str, default='')
parser.add_argument('--test_envscaledir_name', help='envscaledir name (str)', type=str, default='')
parser.add_argument('--test_envscale_niter',   help='envscale number (str)',  type=str, default='')
# train config info
parser.add_argument('--cfgfile_name',    help='cfgfile name (str)',  type=str,     default=task_path + "/cfg.yaml")
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

if args.mode == "train" or args.mode == "retrain":
    print("(re)train mode")
    cfg['environment']['num_envs']          = args.num_envs
    cfg['environment']['num_threads']       = args.num_threads
    cfg['environment']['simulation_dt']     = args.simulation_dt
    cfg['environment']['control_dt']        = args.control_dt
    cfg['environment']['max_time']          = args.max_time
    cfg['environment']['learning_rate']     = args.learning_rate
    cfg['environment']['total_timesteps']   = args.total_timesteps
    cfg['architecture']['policy_net']       = args.policy_net
    cfg['architecture']['value_net']        = args.value_net

if args.mode == "test":
    print("test mode")
    cfg['environment']['num_envs']          = 1
    cfg['environment']['num_threads']       = 1
    cfg['environment']['eval_every_n']      = 1
    cfg['environment']['simulation_dt']     = args.simulation_dt
    cfg['environment']['control_dt']        = args.control_dt
    cfg['environment']['total_timesteps']   = args.total_timesteps
    if args.test_modelfile_name == '':
        raise RuntimeError("--test_modelfile_name is empty")
    if args.test_envscaledir_name == '':
        raise RuntimeError("--test_envscaledir_name is empty")
    if args.test_envscale_niter == '':
        raise RuntimeError("--test_envscale_niter is empty")

# create environment from the configuration file
env = VecEnv(impl=rsg_mingi.RaisimGymEnv(task_path + "/rsc",
                                     dump(cfg['environment'],
                                     Dumper=RoundTripDumper)),
             cfg=cfg['environment'],
             normalize_ob=True,
             seed=0,
             normalize_rew=True,  # not implemented
             clip_obs=10.)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps_per_update = n_steps * env.num_envs
total_update = int(cfg['environment']['total_timesteps'] / total_steps_per_update)
print("steps/update : ", total_steps_per_update, ", n_update : ", total_update)

if args.mode == "train" or args.mode == "retrain":
    # shortcuts
    ob_dim = env.num_obs
    act_dim = env.num_acts

    # save the configuration and other files
    saver = ConfigurationSaver(log_dir=home_path + "/data/runnerlog",
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

    avg_rewards = []

    actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'],
                                            nn.LeakyReLU,
                                            ob_dim,
                                            act_dim),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                             'cuda')

    critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'],
                                              nn.LeakyReLU,
                                              ob_dim,
                                              1),
                               'cuda')

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
                  device='cuda',
                  mini_batch_sampling='shuffle',
                  log_intervals=10)

    extraInfoNames = list()

    for update in range(total_update):
        start = time.time()

        # reset all environments
        # env.reset()   # by skoo because we will use the extrainfo
        env.reset_and_update_info()

        if update % cfg['environment']['eval_every_n'] == 0:
            print("Visualizing and evaluating the current policy")

            # save the trained agenet and env scaling coefficients
            actor.save_deterministic_graph(
                saver.data_dir+"/policy_"+str(update)+'.pt',  # file name
                torch.rand(1, ob_dim).cpu())   # random observation to make the policy graph in torch.jit

            parameters = np.zeros([0], dtype=np.float32)
            for param in actor.deterministic_parameters():
                parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)
            np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)
            env.save_scaling(saver.data_dir, str(update))

            # load the saved data to test it
            loaded_graph = torch.jit.load(saver.data_dir+"/policy_"+str(update)+'.pt')
            env.load_scaling(saver.data_dir, str(update))

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

            # model.save(saver.data_dir+"/policies/policy", update)
            env.save_scaling(saver.data_dir, str(update))
            #env.reset()   # by skoo because we will use the extrainfo
            env.reset_and_update_info()

            # SKOO to get extraInfoNames here ...
            if not extraInfoNames:  # if the list is empty
                infos_keys = list(infos[0].keys())
                for ikey in range(len(infos_keys)):
                    if (infos_keys[ikey] != 'episode') and (infos_keys[ikey] != 'extrainfo'):
                        extraInfoNames.append(infos_keys[ikey])

        # actual training
        reward_sum = 0
        done_sum = 0
        average_dones = 0.

        count_done = 0
        sum_length = 0
        count_pass = 0
        sum_length_pass = 0

        full_episode_survival_check = np.ones(env.num_envs, dtype=int)
        rewards_ep = [[] for _ in range(env.num_envs)]
        extrainfos_ep = [[[] for _ in range(len(extraInfoNames))] for _ in range(env.num_envs)]

        for step in range(n_steps):
            obs = env.observe()
            action = ppo.observe(obs)
            rewards, dones, infos = env.step(action) #infos[ienv]['episode'] has 'reward' and 'length' if it is done

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

            # collect roll-out information per step for ppo update
            ppo.step(value_obs=obs, rews=rewards, dones=dones, infos=infos)

            done_sum = done_sum + sum(dones)
            reward_sum = reward_sum + sum(rewards)

        # take st step to get value obs
        obs = env.observe()
        ppo.update(actor_obs=obs,
                   value_obs=obs,
                   log_this_iteration=update % 10 == 0,
                   update=update)

        end = time.time()

        average_performance = reward_sum / total_steps_per_update
        average_dones = done_sum / total_steps_per_update
        avg_rewards.append(average_performance)

        actor.distribution.enforce_minimum_std((torch.ones(25)*0.2).to('cuda'))

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average per step reward: ", '{:0.10f}'.format(average_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_per_update / (end - start))))

        print('{:<40} {:>6}'.format("count done: ", '{:6.0f}'.format(count_done)))
        if count_done > 0:
            print('{:<40} {:>6}'.format("average done length: ", '{:6.0f}'.format(sum_length / count_done)))
        print('{:<40} {:>6}'.format("count pass: ", '{:6.0f}'.format(count_pass)))
        if count_pass > 0:
            print('{:<40} {:>6}'.format("average pass length: ", '{:6.0f}'.format(sum_length_pass / count_pass)))

        print('std: ')
        print(np.exp(actor.distribution.std.cpu().detach().numpy()))
        print('----------------------------------------------------\n')

else:
    print("loading policy file : ", args.test_modelfile_name)
    ppo_model = torch.jit.load(args.test_modelfile_name)
    print("loading env scaling file : ", os.path.join(args.test_envscaledir_name, "mean_var_" + args.test_envscale_niter + ".csv"))
    env.load_scaling(args.test_envscaledir_name, args.test_envscale_niter)

    sum_episode_reward = 0
    sum_episode_steps = 0

    for iepisode in range(total_update):
        env.reset()
        reward_sum = 0.0
        c_step = 0

        while True:
            c_step += 1
            time.sleep(0.01)
            obs = env.observe(update_mean=False)
            action_deterministic = ppo_model(torch.from_numpy(obs).cpu())
            reward, done, info = env.step(action_deterministic.cpu().detach().numpy())
            reward_sum += reward[0]
            if c_step >= n_steps or done == True:
                print("episode ", iepisode, ", steps : ", c_step, ", reward : ", reward_sum)
                sum_episode_reward += reward_sum
                sum_episode_steps += c_step
                break
    print("average ep steps : {}, average ep reward : {}".format(sum_episode_steps/total_update, sum_episode_reward/total_update))
