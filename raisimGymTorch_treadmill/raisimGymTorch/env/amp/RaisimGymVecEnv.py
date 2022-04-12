# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs

        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()

        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._observation_amp = np.zeros([self.num_envs, 142], dtype=np.float32) # amp mingi
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]

        self._extraInfoNames = self.wrapper.getExtraInfoNames()   # by skoo
        self._extraInfo = np.zeros([self.num_envs, len(self._extraInfoNames)], dtype=np.float32)   # by skoo
        self.extrainfos = [[[] for _ in range(len(self._extraInfoNames))] for _ in range(self.num_envs)]

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_command(self, command):
        self.wrapper.setCommand(command)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done, self._extraInfo)

        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])

            for j in range(len(self._extraInfoNames)):
                self.extrainfos[i][j].append(self._extraInfo[i, j])
                info[i][self._extraInfoNames[j]] = self._extraInfo[i, j]

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"reward": eprew, "length": eplen}
                self.rewards[i].clear()
                info[i]['episode'] = epinfo

                extrainfo = dict()
                for j in range(len(self._extraInfoNames)):
                    epval = sum(self.extrainfos[i][j])/len(self.extrainfos[i][j])
                    extrainfo[self._extraInfoNames[j]] = epval
                    self.extrainfos[i][j].clear()
                info[i]['extrainfo'] = extrainfo

        return self._reward.copy(), self._done.copy(), info.copy()


    def load_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)

        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)

            return self._normalize_observation(self._observation)
        else:
            return self._observation.copy()

    # amp mingi
    def observe_amp(self, update_mean=True):
        self.wrapper.observe_amp(self._observation_amp)

        return self._observation_amp.copy()

    def reset(self):
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.reset()
        # return self.observe()

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"reward": eprew, "length": eplen}
            self.rewards[i].clear()
            info[i]['episode'] = epinfo

            extrainfo = dict()
            for j in range(len(self._extraInfoNames)):
                if len(self.extrainfos[i][j]) > 0:
                    epval = sum(self.extrainfos[i][j])/len(self.extrainfos[i][j])
                else:
                    epval = 0
                extrainfo[self._extraInfoNames[j]] = epval
                self.extrainfos[i][j].clear()
            info[i]['extrainfo'] = extrainfo

        return info

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def extra_info_names(self):
        return self._extraInfoNames


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
