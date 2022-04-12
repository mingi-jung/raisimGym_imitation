import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit
# from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
#import pyqtgraph as pg

import os
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
        self.width = 600
        self.height = 400

        self.controller_weight_filepath = ""
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

        self.controller_weight_filepath = "/home/opensim2020/Desktop/pytorch_result/20210428/2021-04-27-15-58-49/policy_9950.pt"
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

        # graph_draw_knee_angle = pg.PlotWidget(self)
        # graph_draw_knee_angle.move(300, 30)
        # graph_draw_knee_angle.resize(200, 200)

        self.show()


    def controller_filepath_update(self, text):
        # self.controller_weight_filepath = os.path.dirname(os.path.abspath(__file__)) + "/../data/gaitmskdamped/2021-04-27-09-10-47/full_100.pt"
        self.controller_weight_filepath = text


    def load_raisim_do_not_use_it(self):
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
        self.world = raisim.World()
        self.world.setTimeStep(0.001)

        # create objects
        terrainProperties = raisim.TerrainProperties()
        terrainProperties.frequency = 0.2
        terrainProperties.zScale = 3.0
        terrainProperties.xSize = 20.0
        terrainProperties.ySize = 20.0
        terrainProperties.xSamples = 50
        terrainProperties.ySamples = 50
        terrainProperties.fractalOctaves = 3
        terrainProperties.fractalLacunarity = 2.0
        terrainProperties.fractalGain = 0.25
        hm = self.world.addHeightMap(0.0, 0.0, terrainProperties)

        # robot
        anymal_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/urdf/MSK_GAIT2392_model_joint_template.urdf"
        self.anymal = self.world.addArticulatedSystem(anymal_urdf_file)

        # ANYmal joint PD controller
        anymal_nominal_joint_config = np.array([0, -1.5, 2.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8,
                                                -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
        self.anymal.setGeneralizedCoordinate(anymal_nominal_joint_config)
        self.anymal.setPdGains(200 * np.ones([18]), np.ones([18]))
        self.anymal.setPdTarget(anymal_nominal_joint_config, np.zeros([18]))

        # launch raisim server
        self.server = raisim.RaisimServer(self.world)


    def load_raisim_env(self):
        print('load_raisim_env ...... ', end='')
        # create environment from the configuration file
        task_path = os.path.dirname(os.path.abspath(__file__)) + "/../raisimGymTorch/env/envs/rsg_mingi"
        cfg_path = os.path.abspath(task_path + "/cfg.yaml")
        self.cfg = YAML().load(open(cfg_path, 'r'))
        self.cfg['environment']['num_envs'] = 1
        self.cfg['environment']['num_threads'] = 1
        self.cfg['environment']['simulation_dt'] = 0.01
        self.cfg['environment']['control_dt'] = 0.01
        self.env = VecEnv(impl=rsg_mingi.RaisimGymEnv(task_path + "/rsc", dump(self.cfg['environment'], Dumper=RoundTripDumper)),
                     cfg=self.cfg['environment'],
                     normalize_ob=True,
                     seed=0,
                     normalize_rew=True,  # not implemented
                     clip_obs=10.)
        print("num_envs: ", self.env.num_envs, ", num_obs: ", self.env.num_obs, ", num_acts: ", self.env.num_acts, " ...... ", end='')
        print('done')


    @pyqtSlot()
    def on_click_do_not_use_it(self):
        print('PyQt5 button click')
        # launch raisim server
        self.server.launchServer(8080)

        for i in range(500000):
            self.world.integrate()
            time.sleep(0.0005)

        self.server.killServer()


    @pyqtSlot()
    def on_click_load_controller(self):
        print('on_click_load_controller ...... ', end='')
        ob_dim = self.env.num_obs
        act_dim = self.env.num_acts
        weight_path = self.controller_weight_filepath
        self.gait_controller = ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        self.gait_controller.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
        print('done')


    @pyqtSlot()
    def on_click_run_raisim(self):
        print('on_click_run_raisim ......', end='')
        n_steps = 1000
        for step in range(n_steps):
            frame_start = time.time()
            # action = np.array([np.random.uniform(-1.5, 1.5, self.env.num_acts)], dtype=np.float32) # for testing
            # rewards, dones, infos = self.env.step(action)
            obs = self.env.observe(update_mean=False)
            action_deterministic = self.gait_controller.architecture(torch.from_numpy(obs).cpu())
            rewards, dones, infos = self.env.step(action_deterministic.cpu().detach().numpy())
            frame_end = time.time()
            wait_time = self.cfg['environment']['control_dt'] - (frame_end - frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
        print('done')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
