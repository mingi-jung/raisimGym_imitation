import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit
# from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg

import os
import math
import numpy as np
import raisimpy as raisim
import time

from raisimGymTorch.env.bin import rsg_mingi as rsg_mingi
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
# from raisimGymTorch.helper.raisim_gym_helper import load_param
from ruamel.yaml import YAML, dump, RoundTripDumper

import raisimGymTorch.algo.ppo.module as ppo_module
import torch
import torch.nn as nn



class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 50
        self.top = 50
        self.width = 1000
        self.height = 600

        self.controller_weight_filepath = ""
        self.extra_info = ""
        self.initUI()
        # self.load_raisim_do_not_use_it()
        self.load_raisim_env()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # button = QPushButton('PyQt5 button', self)
        # button.setToolTip('This is an example button')
        # button.move(100, 70)
        # button.clicked.connect(self.on_click)

        self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210428/2021-04-27-15-58-49/policy_9950"
        lineedit_controller_filepath = QLineEdit(self.controller_weight_filepath, self)
        # lineedit_controller_filepath.setText("")
        lineedit_controller_filepath.move(100, 30)
        lineedit_controller_filepath.textChanged.connect(self.controller_filepath_update)

        button_load_controller = QPushButton('Load Controller', self)
        button_load_controller.setToolTip('It loads a controller weights')
        button_load_controller.move(100, 70)
        button_load_controller.clicked.connect(self.on_click_load_controller)

        button_run_raisim_env = QPushButton('Run RaisimEnv', self)
        button_run_raisim_env.setToolTip('It runs a RaisimGymVecEnv')
        button_run_raisim_env.move(100, 110)
        button_run_raisim_env.clicked.connect(self.on_click_run_raisim)

        graph_draw_knee_angle = pg.PlotWidget(self)
        graph_draw_knee_angle.move(300, 30)
        graph_draw_knee_angle.resize(600, 500)

        button_run_raisim_env = QPushButton('View Graph', self)
        button_run_raisim_env.setToolTip('It loads a graph')
        button_run_raisim_env.move(100, 190)
        button_run_raisim_env.clicked.connect(self.on_click_run_raisim)

        self.show()


    def controller_filepath_update(self, text):
        # self.controller_weight_filepath = os.path.dirname(os.path.abspath(__file__)) + "/../data/gaitmskdamped/2021-04-27-09-10-47/full_100.pt"
        self.controller_weight_filepath = text

    def load_raisim_env(self):
        print('load_raisim_env ...... ', end='')
        # create environment from the configuration file
        task_path = os.path.dirname(os.path.abspath(__file__)) + "/../raisimGymTorch/env/envs/rsg_mingi"
        cfg_path = os.path.abspath(task_path + "/cfg.yaml")
        self.cfg = YAML().load(open(cfg_path, 'r'))
        self.cfg['environment']['num_envs'] = 1
        self.cfg['environment']['num_threads'] = 1
        self.cfg['environment']['simulation_dt'] = 0.001
        self.cfg['environment']['control_dt'] = 0.01
        self.env = VecEnv(rsg_mingi.RaisimGymEnv(task_path + "/rsc",
                                             dump(self.cfg['environment'],
                                             Dumper=RoundTripDumper)),
                     self.cfg['environment'])
        print("num_envs: ", self.env.num_envs, ", num_obs: ", self.env.num_obs, ", num_acts: ", self.env.num_acts, " ...... ", end='')
        print('done')

    @pyqtSlot()
    def on_click_load_controller(self):
        print('on_click_load_controller ...... ', end='')
        ob_dim = self.env.num_obs
        act_dim = self.env.num_acts
        self.weight_path = self.controller_weight_filepath

        print('done')


    @pyqtSlot()
    def on_click_run_raisim(self):
        print('on_click_run_raisim ......', end='')

        start = time.time()
        self.env.reset()
        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.
        n_steps = math.floor(self.cfg['environment']['max_time'] / self.cfg['environment']['control_dt'])
        total_steps = n_steps * 1
        start_step_id = 0

        print("Visualizing and evaluating the policy", self.weight_path+".pt")
        loaded_graph = torch.jit.load(self.weight_path+'.pt')

        self.env.load_scaling(self.weight_path.rsplit(os.sep, 1)[0], int(self.weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1]))
        print("Load observation scaling in", self.weight_path.rsplit(os.sep, 1)[0]+":", "mean"+str(int(self.weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1])) + ".csv", "and", "var"+str(int(self.weight_path.rsplit(os.sep, 1)[1].split('_', 1)[1])) + ".csv")
        self.env.turn_on_visualization()

        # max_steps = 1000000
        max_steps = 1000 ## 10 secs
        for step in range(max_steps):
            time.sleep(0.01)
            obs = self.env.observe(False)
            action_ll = loaded_graph(torch.from_numpy(obs).cpu())
            reward_ll, dones, infos = self.env.step(action_ll.cpu().detach().numpy())
            reward_ll_sum = reward_ll_sum + reward_ll[0]
            if dones or step == max_steps - 1:
                print('----------------------------------------------------')
                print('{:<40} {:>6}'.format("sum reward: ", '{:0.10f}'.format(reward_ll_sum)))
                print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
                print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
                print('----------------------------------------------------\n')
                start_step_id = step + 1
                reward_ll_sum = 0.0
        self.env.turn_off_visualization()
        self.env.reset()

        print('done')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
