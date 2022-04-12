import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QComboBox, QLabel, QCheckBox
# from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
import pyqtgraph.exporters

import os
import math
import numpy as np
# import raisimpy as raisim
import time

from raisimGymTorch.env.bin import rsg_mingi as rsg_test
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from ruamel.yaml import YAML, dump, RoundTripDumper

import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch
import torch.nn as nn

import json

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 50
        self.top = 50
        self.width = 1000
        self.height = 600

        self.extra_info_value1 = list()
        self.extra_info_value2 = list()

        self.controller_weight_filepath = ""
        self.extra_info1 = ""
        self.extra_info2 = ""
        self.initUI()
        # self.load_raisim_do_not_use_it()
        # self.load_raisim_env()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button_load_env = QPushButton('Load Env', self)
        button_load_env.setToolTip('It loads a env')
        button_load_env.move(100, 30)
        button_load_env.clicked.connect(self.on_click_load_env)

        # initial text, yjkoo 20210803
        self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210611/2021-06-10-15-25-47/full_9950.pt" # normal 100per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210616/2021-06-14-18-03-26/full_9950.pt" # fast 100per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210621/2021-06-18-17-45-14/full_9950.pt" # slow 100per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210622/2021-06-21-17-56-41/full_9950.pt" # run 100per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210702/2021-07-01-20-37-09/full_9950.pt" # run 150per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210623/2021-06-22-17-47-33/full_9950.pt" # normal 75per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210704/2021-07-02-17-30-24/full_9950.pt" # normal 125per
        # self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210629/2021-06-28-23-36-55/full_9950.pt" # normal 150per

        label1 = QLabel('Policy network:', self)
        label1.move(100, 75)

        lineedit_controller_filepath = QLineEdit(self.controller_weight_filepath, self)
        lineedit_controller_filepath.move(100, 90)
        lineedit_controller_filepath.textChanged.connect(self.controller_filepath_update)

        button_load_controller = QPushButton('Load Controller', self)
        button_load_controller.setToolTip('It loads a controller weights')
        button_load_controller.move(100, 110)
        button_load_controller.clicked.connect(self.on_click_load_controller)

        self.graph_draw = pg.PlotWidget(self)
        self.graph_draw.move(300, 30)
        self.graph_draw.resize(600, 500)

        label2 = QLabel('Extra info:', self)
        label2.move(100, 155)

        self.combobox1 = QComboBox(self)
        self.combobox1.move(100, 170)
        self.combobox1.resize(180,20)

        self.check1 = QCheckBox('Red', self)
        self.check1.move(30, 170)
        self.check1.setChecked(True)

        self.combobox2 = QComboBox(self)
        self.combobox2.move(100, 195)
        self.combobox2.resize(180,20)

        self.check2 = QCheckBox('Green', self)
        self.check2.move(30, 195)
        self.check2.setChecked(True)

        self.combobox3 = QComboBox(self)
        self.combobox3.move(100, 220)
        self.combobox3.resize(180,20)

        self.check3 = QCheckBox('Blue', self)
        self.check3.move(30, 220)
        self.check3.setChecked(True)

        button_plot = QPushButton('Plot Extra Info.', self)
        button_plot.move(100, 250)
        button_plot.clicked.connect(self.on_click_plot)

        label3 = QLabel('Step size:', self)
        label3.move(100, 295)

        self.nFrame = 1500
        lineedit_controller_filepath = QLineEdit(str(self.nFrame), self)
        lineedit_controller_filepath.move(100, 310)
        lineedit_controller_filepath.textChanged.connect(self.step_size_update)

        button_run_raisim_env = QPushButton('Run RaisimEnv', self)
        button_run_raisim_env.setToolTip('It runs a RaisimGymVecEnv')
        button_run_raisim_env.move(100, 330)
        button_run_raisim_env.clicked.connect(self.on_click_run_raisim)


        label3 = QLabel('File name:', self)
        label3.move(100, 375)

        self.export_name = "normal_150per_no"
        lineedit_controller_filepath = QLineEdit(self.export_name, self)
        lineedit_controller_filepath.move(100, 390)
        lineedit_controller_filepath.textChanged.connect(self.export_name_update)

        button_save_graph = QPushButton('Save info.', self)
        button_save_graph.setToolTip('It saves graph as a image')
        button_save_graph.move(100, 410)
        button_save_graph.clicked.connect(self.on_click_save_graph)

        self.show()

    def controller_filepath_update(self, text):
        self.controller_weight_filepath = text

    def step_size_update(self, text):
        self.nFrame = int(text)

    def export_name_update(self, text):
        self.export_name = text

    @pyqtSlot()
    def on_click_load_env(self):
        print('load_raisim_env ...... ', end='')

        task_path = os.path.dirname(os.path.abspath(__file__)) + "/../raisimGymTorch/env/envs/rsg_mingi"
        # task_path = os.path.realpath(__file__)
        cfg_path = os.path.abspath(task_path + "/cfg.yaml")
        self.cfg = YAML().load(open(cfg_path, 'r'))

        self.cfg['environment']['num_envs'] = 1
        self.cfg['environment']['num_threads'] = 1
        self.cfg['environment']['simulation_dt'] = 0.001
        self.cfg['environment']['control_dt'] = 0.01

        self.env = VecEnv(rsg_test.RaisimGymEnv(task_path + "/rsc",
                                             dump(self.cfg['environment'],
                                             Dumper=RoundTripDumper)),
                     self.cfg['environment'])

        self.task_path = task_path
        print("num_envs: ", self.env.num_envs, ", num_obs: ", self.env.num_obs, ", num_acts: ", self.env.num_acts, " ...... ", end='')
        print('done')

    def on_click_plot(self):
        x_value1 = list()
        x_value2 = list()
        x_value3 = list()
        y_value1 = list()
        y_value2 = list()
        y_value3 = list()
        self.graph_draw.clear()

        plot_info1 = str(self.combobox1.currentText())
        plot_info2 = str(self.combobox2.currentText())
        plot_info3 = str(self.combobox3.currentText())

        size_x = len(self.extraInfo_frame[plot_info1])

        if self.check1.isChecked():
            for frame in range(size_x):
                x_value1.append(frame*0.01-0.01)
                y_value1.append(self.extraInfo_frame[plot_info1][frame])
            self.graph_draw.plot(x=x_value1, y=y_value1, pen=pg.mkPen('r', width=3))

        if self.check2.isChecked():
            for frame in range(size_x):
                x_value2.append(frame*0.01-0.01)
                y_value2.append(self.extraInfo_frame[plot_info2][frame])
            self.graph_draw.plot(x=x_value2, y=y_value2, pen=pg.mkPen('g', width=3))

        if self.check3.isChecked():
            for frame in range(size_x):
                x_value3.append(frame*0.01-0.01)
                y_value3.append(self.extraInfo_frame[plot_info3][frame])
            self.graph_draw.plot(x=x_value3, y=y_value3, pen=pg.mkPen('b', width=3))


    @pyqtSlot()
    def on_click_load_controller(self):
        print('on_click_load_controller ...... ', end='')
        ob_dim = self.env.num_obs
        act_dim = self.env.num_acts
        self.weight_path = self.controller_weight_filepath

        actor = ppo_module.Actor(ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                                ppo_module.MultivariateGaussianDiagonalCovariance(self.env.num_acts, 1.0),
                                'cpu')

        critic = ppo_module.Critic(ppo_module.MLP(self.cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                                'cpu')

        saver = ConfigurationSaver(log_dir=("/home/opensim2020/raisim_v5_workspace/raisimLib/raisimGymTorch_treadmill/pyqt_model_runner_koo/data/temp"), save_items=[os.path.abspath(self.task_path + "/cfg.yaml"), os.path.abspath(self.task_path + "/Environment.hpp")])

        ppo = PPO.PPO(actor=actor,
                critic=critic,
                num_envs=self.cfg['environment']['num_envs'],
                num_transitions_per_env=100,
                num_learning_epochs=4,
                num_mini_batches=4,
                clip_param=0.2,
                gamma=0.998,
                lam=0.95,
                value_loss_coef=0.5,
                entropy_coef=0.0,
                learning_rate=self.cfg['environment']['learning_rate'],
                max_grad_norm=0.5,
                use_clipped_value_loss=True,
                log_dir=saver.data_dir,
                device='cpu',
                mini_batch_sampling='in_order',
                log_intervals=10)


        load_param(self.controller_weight_filepath, self.env, actor, critic, ppo.optimizer, saver.data_dir)


        self.extraInfoNames = list()
        self.extraInfo_frame = dict()

        obs = self.env.observe(update_mean=False)

        loaded_graph = ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, self.env.num_obs, self.env.num_acts)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        _, _, infos = self.env.step(action_ll.cpu().detach().numpy())
        self.env.reset_and_update_info()

        infos_keys = list(infos[0].keys())
        for ikey in range(len(infos_keys)):
            if (infos_keys[ikey] != 'episode') and (infos_keys[ikey] != 'extrainfo'):
                self.extraInfoNames.append(infos_keys[ikey])
                self.combobox1.addItem(infos_keys[ikey])
                self.combobox2.addItem(infos_keys[ikey])
                self.combobox3.addItem(infos_keys[ikey])

        print('done')


    @pyqtSlot()
    def on_click_run_raisim(self):
        print('on_click_run_raisim ......', end='')

        self.graph_draw.clear()

        start = time.time()
        self.env.reset_and_update_info()
        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.
        n_steps = math.floor(self.cfg['environment']['max_time'] / self.cfg['environment']['control_dt'])
        total_steps = n_steps * 1
        start_step_id = 0

        print("Visualizing and evaluating the policy", self.weight_path+".pt")
        self.env.reset_and_update_info()
        loaded_graph = ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, self.env.num_obs, self.env.num_acts)
        loaded_graph.load_state_dict(torch.load(self.weight_path)['actor_architecture_state_dict'])

        self.env.turn_on_visualization()

        self.json_path = "./" + self.export_name + ".json"
        self.data = {}

        # max_steps = 1000000
        max_steps = self.nFrame
        for step in range(max_steps):
            frame_start = time.time()
            obs = self.env.observe(update_mean=False)
            action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward_ll, dones, infos = self.env.step(action_ll.cpu().detach().numpy())
            frame_end = time.time()
            wait_time = 0.01 - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

            reward_ll_sum = reward_ll_sum + reward_ll[0]
            if dones or step == max_steps - 1:
                print('----------------------------------------------------')
                print('{:<40} {:>6}'.format("sum reward: ", '{:0.10f}'.format(reward_ll_sum)))
                print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
                print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
                print('----------------------------------------------------\n')
                start_step_id = step + 1
                reward_ll_sum = 0.0

            for j in range(len(self.extraInfoNames)):
                if step == 0:
                    self.extraInfo_frame[self.extraInfoNames[j]] = list()
                    self.extraInfo_frame[self.extraInfoNames[j]].clear()

                self.extraInfo_frame[self.extraInfoNames[j]].append(infos[0][self.extraInfoNames[j]])

        self.env.turn_off_visualization()
        self.env.reset_and_update_info()

        print('done')

    @pyqtSlot()
    def on_click_save_graph(self):
        print('on_click_save_graph ...... ', end='')

        exporter = pg.exporters.ImageExporter(self.graph_draw.plotItem)
        exporter.parameters()['width'] = 1200
        exporter.export(self.export_name + '.png')

        self.data = {}

        for n_info in range(len(self.extraInfoNames)):
            self.data[self.extraInfoNames[n_info]] = list()
            for n_frame in range(len(self.extraInfo_frame[str(self.combobox1.currentText())])):
                self.data[self.extraInfoNames[n_info]].append(float(self.extraInfo_frame[self.extraInfoNames[n_info]][n_frame]))

        with open(self.json_path, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

        print('done')




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
