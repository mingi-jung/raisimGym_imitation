from ruamel.yaml import YAML, dump, RoundTripDumper

from raisimGymTorch.env.bin import rsg_gaitmsk_amp as rsg_anymal
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param

import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import AMPDisc

import os
import math
import time
import argparse

import torch
import torch.nn as nn
import numpy as np

import psutil
# import thingspeak
#
# thingspeak_channel_id = "1270590"  # DRL2 log channel
# thingspeak_write_key = "CL27IW7KOPJWWCBC___"  # DRL2 write key
# thingspeak_channel = thingspeak.Channel(id=thingspeak_channel_id, api_key=thingspeak_write_key)
#
# def log_thingspeak(channel, reward_per_step):
#     cpu_pc = psutil.cpu_percent()
#     mem_avail_mb = psutil.virtual_memory().percent
#
#     try:
#         response = channel.update({"field1": cpu_pc, "field2":mem_avail_mb, "field3":reward_per_step})
#         # print(cpu_pc)
#         # print(mem_avail_mb)
#         print(response)
#     except:
#         print("connection failed")


# task specification
task_name = "gaitmsk_amp"

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = os.path.abspath(task_path + "/../../../..")

# config
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
parser.add_argument('-wp', '--ppo_weight', help='pre-trained ppo weight path', type=str, default='')
parser.add_argument('-wd', '--disc_weight', help='pre-trained discriminator weight path', type=str, default='')
parser.add_argument('-c', '--cfgfile_name', help='cfgfile name (str)', type=str,
                    default=os.path.abspath(task_path + "/cfg.yaml"))

args = parser.parse_args()
cfg = YAML().load(open(args.cfgfile_name, 'r'))

mode = args.mode
ppo_weight_path = args.ppo_weight
disc_weight_path = args.disc_weight

# curriculum_phase = cfg['environment']['curriculum_phase']
# curriculum_start = cfg['environment']['curriculum_start']
# curriculum_call_init = cfg['environment']['curriculum_call_init']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(mode, "mode")
print("cuda") if torch.cuda.is_available() else print("cpu")
# print("curriculum starts at ", curriculum_start) if curriculum_phase == 1 else None
# print("curriculum update initial call count is ", curriculum_call_init) if curriculum_phase == 1 else None

# create environment from the configuration file
env = VecEnv(impl=rsg_anymal.RaisimGymEnv(task_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg=cfg['environment'],
             normalize_ob=True,
             seed=0,
             normalize_rew=True,  # not implemented
             clip_obs=10.)

# shortcuts
ob_dim = env.num_obs
ob_amp_dim = env.num_obs_amp
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
n_steps_per_update = n_steps * env.num_envs
n_update = int(cfg['environment']['total_timesteps'] / n_steps_per_update)

print("ob_dim : ", ob_dim)
print("act_dim : ", act_dim)
print("n_update : ", n_update)
print("steps per episode : ", n_steps)
print("steps/update : ", n_steps_per_update)

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=os.path.abspath(home_path + "/data/" + task_name),
                           save_items=[os.path.abspath(task_path + "/cfg.yaml"),
                                       os.path.abspath(task_path + "/Environment.hpp"),
                                       os.path.abspath(task_path + "/Environment_deepmimic.hpp"),
                                       os.path.abspath(task_path + "/Environment_muscle.hpp"),
                                       os.path.abspath(task_path + "/DeepmimicUtility.hpp"),
                                       os.path.abspath(task_path + "/Muscle.hpp"),
                                       os.path.abspath(task_path + "/quaternion_mskbiodyn.hpp"),
                                       os.path.abspath(task_path + "/AMPDisc.py"),
                                       os.path.abspath(task_path + "/runnerMod_train.py")])

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
              betas=(0.9, 0.999),
              max_grad_norm=0.5,
              use_clipped_value_loss=True,
              log_dir=saver.data_dir,
              device=device,
              mini_batch_sampling='in_order',
              log_intervals=10)

disc = AMPDisc.AMPDisc(ndim=ob_amp_dim*2, num_envs=cfg['environment']['num_envs'], device=device, learning_rate=1e-5, betas=(0.5, 0.999))
# disc.load_refmotion_data(task_path + "/rsc/refmotion/" + cfg['environment']['expert_motion_sample'])
disc.load_refmotion_data(home_path + "/raisimGymTorch/env/envs/rsg_gaitmsk_amp/rsc/expert/" + cfg['environment']['expert_motion_sample'])

if mode == 'retrain':
    if ppo_weight_path != '':
        load_param(ppo_weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    if disc_weight_path != '':
        disc.load_state(disc_weight_path)

# Obtain extraInfoNames
extraInfoNames = list()
obs = env.observe()
action = ppo.observe(obs)
_, _, infos = env.step(action)
infos_keys = list(infos[0].keys())
for ikey in range(len(infos_keys)):
    if (infos_keys[ikey] != 'episode') and (infos_keys[ikey] != 'extrainfo'):
        extraInfoNames.append(infos_keys[ikey])

disc_update_skip = 5
d_loss = 0.0
d_loss_agent = 0.0
d_loss_expert = 0.0
d_gradient_penalty = 0.0

for update in range(n_update):
    start = time.time()
    env.reset_and_update_info()

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        env.save_scaling(saver.data_dir, str(update))
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict()},
            os.path.abspath(saver.data_dir + "/full_" + str(update) + '.pt'))

        disc.save_state(os.path.abspath(saver.data_dir + "/full_disc_" + str(update) + '.pt'))

        # we create another graph just to demonstrate the save/load method
        # loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        # loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])
        #
        # env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"policy_"+str(update)+'.mp4')
        #
        # infos = list()
        # for step in range(n_steps*2):
        #     frame_start = time.time()
        #     obs = env.observe(update_mean=False) # observation normalization is on
        #     # action_deterministic = loaded_graph(torch.from_numpy(obs).cpu())
        #     action_deterministic = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        #     _, _, infos = env.step(action_deterministic.cpu().detach().numpy())
        #     frame_end = time.time()
        #     wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        #     if wait_time > 0.:
        #         time.sleep(wait_time)
        #
        # env.stop_video_recording()
        # env.turn_off_visualization()
        # env.reset_and_update_info()

    # actual training
    reward_sum = 0.
    done_sum = 0.
    average_dones = 0.

    count_done = 0.
    sum_length = 0.
    count_pass = 0.

    full_episode_survival_check = np.ones(env.num_envs, dtype=int)

    rewards_ep = [[] for _ in range(env.num_envs)]
    extrainfos_ep = [[[] for _ in range(len(extraInfoNames))] for _ in range(env.num_envs)]
    ampinfos_ep = [[[] for _ in range(8)] for _ in range(env.num_envs)]

    time_step_start = time.time()

    for step in range(n_steps):
        obs = env.observe()  # observation normalization is on
        obs_amp_0 = env.observe_amp()
        action = ppo.observe(obs)
        rewards_task, dones, infos = env.step(action)
        obs_amp_1 = env.observe_amp()
        disc_score = disc.observe(obs_amp_0, obs_amp_1)
        # rewards_style = np.exp(-1.0 * 0.7*(1-disc_score)**2)
        rewards_style = np.maximum(0, 1 - 0.25 * (disc_score - 1)**2)
        # print(rewards_style)

        rewards_style_raw = np.abs(1-disc_score)

        # 20210804:1806, Okay now we train AMP
        rewards = 0.15*rewards_task + 0.85*rewards_style
        # rewards = rewards_task

        # update "infos" for tensorboard log
        for ienv in range(env.num_envs):
            if infos[ienv].get('extrainfo') is not None:
                infos[ienv].pop('extrainfo')  # remove the extrainfo in the infos. It is taken care of in a different way.
            rewards_ep[ienv].append(rewards[ienv])
            for j in range(len(extraInfoNames)):
                extrainfos_ep[ienv][j].append(infos[ienv][extraInfoNames[j]])
            ampinfos_ep[ienv][0].append(rewards[ienv])
            ampinfos_ep[ienv][1].append(rewards_task[ienv])
            ampinfos_ep[ienv][2].append(rewards_style[ienv])
            ampinfos_ep[ienv][3].append(d_loss)
            ampinfos_ep[ienv][4].append(d_loss_agent)
            ampinfos_ep[ienv][5].append(d_loss_expert)
            ampinfos_ep[ienv][6].append(d_gradient_penalty)
            ampinfos_ep[ienv][7].append(rewards_style_raw[ienv])

        for ienv in range(env.num_envs):
            if dones[ienv] == 1:
                # the 'episode' and 'extrainfo' are entered into the tensorboard log
                full_episode_survival_check[ienv] = 0
                infos[ienv]['episode']['reward'] = sum(rewards_ep[ienv])  # it is available when done==1
                infos[ienv]['episode']['reward_per_step'] = infos[ienv]['episode']['reward'] / infos[ienv]['episode']['length']  # append "reward_per_step"

                extrainfo = dict()
                for j in range(len(extraInfoNames)):
                    extrainfo[extraInfoNames[j]] = sum(extrainfos_ep[ienv][j]) / infos[ienv]['episode']['length']
                extrainfo['reward_total'] = sum(ampinfos_ep[ienv][0]) / infos[ienv]['episode']['length']
                extrainfo['reward_task'] = sum(ampinfos_ep[ienv][1]) / infos[ienv]['episode']['length']
                extrainfo['reward_style'] = sum(ampinfos_ep[ienv][2]) / infos[ienv]['episode']['length']
                extrainfo['disc_loss'] = sum(ampinfos_ep[ienv][3]) / infos[ienv]['episode']['length']
                extrainfo['disc_loss_agent'] = sum(ampinfos_ep[ienv][4]) / infos[ienv]['episode']['length']
                extrainfo['disc_loss_expert'] = sum(ampinfos_ep[ienv][5]) / infos[ienv]['episode']['length']
                extrainfo['disc_gradient_penalty'] = sum(ampinfos_ep[ienv][6]) / infos[ienv]['episode']['length']
                extrainfo['reward_style_raw'] = sum(ampinfos_ep[ienv][7]) / infos[ienv]['episode']['length']
                infos[ienv]['extrainfo'] = extrainfo  # update extrainfo infos

                count_done += 1
                sum_length += infos[ienv]['episode']['length']

                rewards_ep[ienv].clear()
                ampinfos_ep[ienv][0].clear()
                ampinfos_ep[ienv][1].clear()
                ampinfos_ep[ienv][2].clear()
                ampinfos_ep[ienv][3].clear()
                ampinfos_ep[ienv][4].clear()
                ampinfos_ep[ienv][5].clear()
                ampinfos_ep[ienv][6].clear()
                ampinfos_ep[ienv][7].clear()

        if step == n_steps - 1:   # in case of the last step
            for ienv in range(env.num_envs):
                if full_episode_survival_check[ienv] != 1:
                    # do not count the envs that has fallen and restarted
                    continue
                count_pass += 1
                eprew = sum(rewards_ep[ienv])  # sum of a episode reward
                eplen = len(rewards_ep[ienv])  # len of a episode
                epsteprew = eprew / eplen  # avg step reward of a episode
                epinfo = {"reward": eprew, "length": eplen, "reward_per_step": epsteprew}
                infos[ienv]['episode'] = epinfo  # update episode infos

                extrainfo = dict()
                for j in range(len(extraInfoNames)):
                    extrainfo[extraInfoNames[j]] = sum(extrainfos_ep[ienv][j]) / n_steps
                extrainfo['reward_total'] = sum(ampinfos_ep[ienv][0]) / n_steps
                extrainfo['reward_task'] = sum(ampinfos_ep[ienv][1]) / n_steps
                extrainfo['reward_style'] = sum(ampinfos_ep[ienv][2]) / n_steps
                extrainfo['disc_loss'] = sum(ampinfos_ep[ienv][3]) / n_steps
                extrainfo['disc_loss_agent'] = sum(ampinfos_ep[ienv][4]) / n_steps
                extrainfo['disc_loss_expert'] = sum(ampinfos_ep[ienv][5]) / n_steps
                extrainfo['disc_gradient_penalty'] = sum(ampinfos_ep[ienv][6]) / n_steps
                extrainfo['reward_style_raw'] = sum(ampinfos_ep[ienv][7]) / n_steps
                infos[ienv]['extrainfo'] = extrainfo  # update extrainfo infos

        # collect roll-out information per step for ppo update
        ppo.step(value_obs=obs, rews=rewards, dones=dones, infos=infos)
        done_sum = done_sum + sum(dones)
        reward_sum = reward_sum + sum(rewards)

    time_step_end_update_start = time.time()

    # take st step to get value obs
    obs = env.observe()  # observation normalization is on
    ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)

    # slow down the disc training
    if update % 1000 == 0:
        disc_update_skip = disc_update_skip + 1
    if update % disc_update_skip == 0:
        d_loss, d_loss_agent, d_loss_expert, d_gradient_penalty = disc.update()
    disc.agent_data.clear()

    actor.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    average_performance = reward_sum / n_steps_per_update
    average_dones = done_sum / n_steps_per_update
    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average reward per step: ", '{:0.10f}'.format(average_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(n_steps_per_update / (end - start))))
    print('{:<40} {:>6}'.format("time elapsed in step loop: ",
                                '{:6.4f}'.format(time_step_end_update_start - time_step_start)))
    print('{:<40} {:>6}'.format("time elapsed in ppo update: ", '{:6.4f}'.format(end - time_step_end_update_start)))
    print('{:<40} {:>6}'.format("count done: ", '{:6.0f}'.format(count_done)))
    if count_done > 0: # envs that fell in the middle
        print('{:<40} {:>6}'.format("average done length: ", '{:6.0f}'.format(sum_length / count_done)))
    print('{:<40} {:>6}'.format("count pass: ", '{:6.0f}'.format(count_pass)))
    # if count_pass > 0:
    #     print('{:<40} {:>6}'.format("average pass length: ", '{:6.0f}'.format(sum_length_pass / count_pass)))

    # print('std: ')
    # print(np.exp(actor.distribution.std.cpu().detach().numpy()))

    # log to thingspeak channel
    # if update % 20 == 0:
    #     log_thingspeak(thingspeak_channel, average_performance)
    print('----------------------------------------------------\n')
